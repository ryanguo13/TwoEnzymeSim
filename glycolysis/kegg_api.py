from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests
import xml.etree.ElementTree as ET
import math

try:
    import tomllib  # Python 3.11+
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore

# eQuilibrator (optional)
try:
    from equilibrator_api import ComponentContribution, Reaction, Q_  # type: ignore
    HAS_EQ = True
except Exception:
    HAS_EQ = False


KEGG_KGML_CANDIDATES = [
    # Reference pathway
    "http://rest.kegg.jp/get/map00010/kgml",
    "https://rest.kegg.jp/get/map00010/kgml",
    # Explicit path prefix
    "http://rest.kegg.jp/get/path:map00010/kgml",
    "https://rest.kegg.jp/get/path:map00010/kgml",
    # Common organism examples (fallbacks). Users can replace with their organism.
    "http://rest.kegg.jp/get/hsa00010/kgml",
    "https://rest.kegg.jp/get/hsa00010/kgml",
]


def fetch_kgml(url: str, timeout: float = 15.0, retries: int = 3, backoff: float = 1.5) -> str:
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            text = resp.text.strip()
            if not text or not text.startswith("<"):
                raise ValueError("Empty or invalid KGML content returned")
            return text
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < retries:
                time.sleep(backoff ** attempt)
    assert last_err is not None
    raise RuntimeError(f"Failed to download KGML from {url}: {last_err}")


def fetch_kgml_with_fallbacks(candidates: list[str]) -> str:
    errors = []
    for url in candidates:
        try:
            return fetch_kgml(url)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{url} -> {e}")
    raise RuntimeError("All KEGG endpoints failed:\n" + "\n".join(errors))


def parse_kgml_text(xml_text: str) -> ET.Element:
    try:
        root = ET.fromstring(xml_text)
        return root
    except ET.ParseError as e:
        snippet = xml_text[:200].replace("\n", " ")
        raise RuntimeError(f"KGML parse error: {e}. First 200 chars: {snippet}") from e


def load_or_download_kgml(cache_path: Path, force_refresh: bool = False) -> ET.Element:
    if cache_path.exists() and not force_refresh:
        try:
            text = cache_path.read_text(encoding="utf-8").strip()
            if text:
                return parse_kgml_text(text)
        except Exception:
            pass
    xml_text = fetch_kgml_with_fallbacks(KEGG_KGML_CANDIDATES)
    # Only write when we have verified content
    cache_path.write_text(xml_text, encoding="utf-8")
    return parse_kgml_text(xml_text)


def extract_reactions_and_compounds(root: ET.Element):
    reactions = []
    compounds = []
    for elem in root.findall(".//reaction"):
        reaction_id = elem.get("id")
        name = elem.get("name")
        substrates = [sub.get("id") for sub in elem.findall(".//substrate") if sub.get("id")]
        products = [prod.get("id") for prod in elem.findall(".//product") if prod.get("id")]
        reactions.append({
            "id": reaction_id,
            "name": name,
            "substrates": substrates,
            "products": products,
        })

    for entry in root.findall(".//entry"):
        if entry.get("type") == "compound":
            cid = entry.get("id")
            cname = entry.get("name")
            compounds.append({"id": cid, "name": cname})

    return reactions, compounds


# Canonical EC numbers for the 10 glycolysis steps (Embden-Meyerhof-Parnas)
GLYCOLYSIS_STEP_EC = {
    "step1": "2.7.1.1",    # Hexokinase
    "step2": "5.3.1.9",    # Phosphoglucose isomerase
    "step3": "2.7.1.11",   # Phosphofructokinase
    "step4": "4.1.2.13",   # Aldolase
    "step5": "5.3.1.1",    # Triose-phosphate isomerase
    "step6": "1.2.1.12",   # GAPDH
    "step7": "2.7.2.3",    # Phosphoglycerate kinase
    "step8": "5.4.2.1",    # Phosphoglycerate mutase (2,3-BPG dependent)
    "step9": "4.2.1.11",   # Enolase
    "step10": "2.7.1.40",  # Pyruvate kinase
}


# Default reaction formulas for the 10 steps (names recognized by eQuilibrator)
# These are overall stoichiometries without compartments.
GLYCOLYSIS_REACTION_FORMULAS = {
    "step1": "glucose + atp <=> g6p + adp + h",
    "step2": "g6p <=> f6p",
    "step3": "f6p + atp <=> fdp + adp + h",
    "step4": "fdp <=> dhap + g3p",
    "step5": "dhap <=> g3p",
    "step6": "g3p + nad + pi <=> 13dpg + nadh + h",
    "step7": "13dpg + adp <=> 3pg + atp",
    "step8": "3pg <=> 2pg",
    "step9": "2pg <=> pep + h2o",
    "step10": "pep + adp + h <=> pyruvate + atp",
}


def compute_dgprime_equilibrator(step: str, formula: str, T: float, pH: float, ionic_strength: float, pMg: float) -> float:
    if not HAS_EQ:
        raise RuntimeError("equilibrator_api is not available. Install equilibrator-api to use --use-equilibrator.")
    cc = ComponentContribution()
    rxn = Reaction.parse_formula(formula)
    # Conditions use pint quantities with eQuilibrator's Q_
    dGp = cc.standard_dg_prime(
        rxn,
        temperature=Q_(T, 'K'),
        p_h=pH,
        ionic_strength=Q_(ionic_strength, 'M'),
        p_mg=pMg,
    )
    # dGp is a Quantity with units of kJ/mol
    return float(dGp.magnitude)


def fetch_sabio_kinetics_for_ec(ec_number: str, timeout: float = 20.0) -> list[float]:
    """
    Fetch kinetic constants from SABIO-RK for a given EC number.

    NOTE: SABIO-RK stores kcat, Km, and sometimes forward rate constants. As a pragmatic
    approximation for mass-action kf, we use available kcat values (1/s) across entries.
    Returns a list of numeric values; caller may select median as representative.
    """
    # Minimalistic query. See SABIO-RK REST docs for richer filtering.
    base = "https://sabiork.h-its.org/sabioRestWebServices/kineticLaws"
    params = {"ecNumber": ec_number}
    headers = {"Accept": "application/xml"}
    try:
        resp = requests.get(base, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        xml_text = resp.text
        # Quick parse: extract <parameter> with name containing 'kcat' or 'kcat (1/s)'
        root = ET.fromstring(xml_text)
        values: list[float] = []
        for param in root.findall('.//parameter'):
            pname = (param.get('name') or '').lower()
            unit = (param.get('unit') or '').lower()
            val = param.get('value') or ''
            if not val:
                continue
            try:
                x = float(val)
            except Exception:
                continue
            # Prefer kcat; accept forward rate constants if clearly marked
            if 'kcat' in pname and (unit in ('1/s', 's^-1', '1 sec^-1', 'per second') or unit == '' or unit is None):
                values.append(x)
            elif ('kforward' in pname or 'k_forward' in pname or 'forward rate constant' in pname) and unit:
                values.append(x)
        return values
    except Exception:
        return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Glycolysis helper: fetch KEGG KGML and/or report kf/kr & ΔG from TOML")
    parser.add_argument("--refresh", action="store_true", help="Force refresh the cached KGML file")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent / "deltaG.toml"),
                        help="Path to deltaG.toml configuration")
    parser.add_argument("--format", type=str, choices=["table", "json", "csv"], default="table",
                        help="Output format for kf/kr/ΔG report")
    parser.add_argument("--temperature", type=float, default=298.15, help="Temperature in Kelvin for Keq computation")
    parser.add_argument("--no-kegg", action="store_true", help="Skip KEGG fetch/parse step")
    parser.add_argument("--use-sabio", action="store_true", help="Attempt to fetch kf from SABIO-RK via EC numbers")
    parser.add_argument("--use-equilibrator", action="store_true", help="Use eQuilibrator to compute ΔG' (non-standard)")
    parser.add_argument("--strict-equilibrator", action="store_true", help="Fail if eQuilibrator ΔG' cannot be computed (no fallback to TOML)")
    parser.add_argument("--ph", type=float, default=7.0, help="pH for eQuilibrator ΔG' computation")
    parser.add_argument("--ionic-strength", type=float, default=0.25, help="Ionic strength (M) for eQuilibrator ΔG'")
    parser.add_argument("--pmg", type=float, default=3.0, help="pMg for eQuilibrator ΔG'")
    parser.add_argument("--write-toml", type=str, default="params_kegg_sabio_eq.toml", help="Write results to TOML at given path")
    parser.add_argument("--kf-source", type=str, choices=["sabio", "baseline", "toml"], default="sabio",
                        help="Source for kf: sabio (preferred), baseline (constant), or toml")
    parser.add_argument("--baseline-kf", type=float, default=1.0, help="Baseline kf (1/s) when kf-source=baseline or SABIO missing")
    args = parser.parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    cache_path = script_dir / "map00010.xml"

    try:
        if not args.no_kegg:
            root = load_or_download_kgml(cache_path, force_refresh=args.refresh)
            reactions, compounds = extract_reactions_and_compounds(root)

            print("Reactions:")
            for r in reactions:
                print(f"Reaction {r['id']} ({r['name']}):")
                print(f"  Substrates: {', '.join(r['substrates'])}")
                print(f"  Products: {', '.join(r['products'])}")

            print("\nCompounds:")
            for c in compounds:
                print(f"Compound {c['id']}: {c['name']}")

        # Always try to produce the kf/kr/ΔG report if tomllib is available
        if tomllib is None:
            print("\n⚠️ tomllib not available; skipping TOML report.")
            return 0

        cfg_path = Path(args.config)
        if not cfg_path.exists():
            # If using equilibrator only, we can proceed without local TOML
            if args.use_equilibrator:
                cfg = {}
            else:
                raise FileNotFoundError(f"Config TOML not found: {cfg_path}")
        else:
            with cfg_path.open("rb") as f:
                cfg = tomllib.load(f)

        deltaG = cfg.get("deltaG", {})
        step_keys = [f"step{i}" for i in range(1, 11)]
        # gas constant in kJ/mol/K to match ΔG units in TOML
        R = 8.314e-3
        T = float(args.temperature)

        report = []
        for i, step in enumerate(step_keys, start=1):
            ksec = cfg.get(f"k{i}", {})
            kf = float(ksec.get("kf", float("nan")))
            kr = float(ksec.get("kr", float("nan")))
            # Determine ΔG: prefer eQuilibrator if enabled; fallback to TOML deltaG
            dg_source = "toml"
            if args.use_equilibrator:
                try:
                    formula = GLYCOLYSIS_REACTION_FORMULAS[step]
                    dg = compute_dgprime_equilibrator(step, formula, T=T, pH=args.ph, ionic_strength=args.ionic_strength, pMg=args.pmg)
                    dg_source = "equilibrator"
                except Exception as e_eq:
                    if args.strict_equilibrator:
                        raise RuntimeError(f"Failed to compute ΔG' for {step} via eQuilibrator: {e_eq}") from e_eq
                    else:
                        dg = float(deltaG.get(step, float("nan")))
                        dg_source = "toml_fallback"
            else:
                dg = float(deltaG.get(step, float("nan")))
                dg_source = "toml"
            keq = math.exp(-dg / (R * T)) if (not math.isnan(dg)) else float("nan")

            # kf source policy
            ec = GLYCOLYSIS_STEP_EC.get(step)
            sabio_values: list[float] = []
            sabio_kf = float('nan')
            kf_source_used = args.kf_source
            if args.kf_source == "sabio":
                if args.use_sabio and ec is not None:
                    sabio_values = fetch_sabio_kinetics_for_ec(ec)
                    if sabio_values:
                        sv = sorted(sabio_values)
                        mid = len(sv) // 2
                        sabio_kf = sv[mid] if (len(sv) % 2 == 1) else 0.5 * (sv[mid - 1] + sv[mid])
                        kf = sabio_kf
                    else:
                        kf = float(args.baseline_kf)
                        kf_source_used = "baseline"
                else:
                    kf = float(args.baseline_kf)
                    kf_source_used = "baseline"
                kr = (kf / keq) if (not math.isnan(keq) and keq != 0.0) else float("nan")
            elif args.kf_source == "baseline":
                kf = float(args.baseline_kf)
                kf_source_used = "baseline"
                kr = (kf / keq) if (not math.isnan(keq) and keq != 0.0) else float("nan")
            else:
                # toml: leave kf/kr as-is
                kf_source_used = "toml"

            ratio = (kf / kr) if (not math.isnan(kf) and not math.isnan(kr) and kr != 0.0) else float("nan")
            discrepancy = (ratio / keq) if (not math.isnan(ratio) and not math.isnan(keq) and keq != 0.0) else float("nan")
            report.append({
                "step": step,
                "EC": ec,
                "kf": kf,
                "kr": kr,
                "deltaG_kJ_per_mol": dg,
                "deltaG_source": dg_source,
                "Keq": keq,
                "kf_over_kr": ratio,
                "ratio_over_Keq": discrepancy,
                "sabio_n": len(sabio_values),
                "kf_source": kf_source_used,
            })

        fmt = args.format
        if fmt == "json":
            import json
            print("\nKF/KR/ΔG Report (JSON):")
            print(json.dumps(report, indent=2))
        elif fmt == "csv":
            import csv
            import io
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=list(report[0].keys()))
            writer.writeheader()
            for row in report:
                writer.writerow(row)
            print("\nKF/KR/ΔG Report (CSV):")
            print(buf.getvalue().strip())
        else:
            # table
            print("\nKF/KR/ΔG Report (T=%.2f K):" % T)
            header = (
                f"{'step':<8}{'EC':<12}{'kf':>12}{'kr':>12}{'ΔG(kJ/mol)':>14}{'Keq':>16}{'kf/kr':>16}{'ratio/Keq':>16}{'SABIO n':>10}{'kf_source':>12}"
            )
            print(header)
            for row in report:
                print(
                    f"{row['step']:<8}"
                    f"{(row['EC'] or ''):<12}"
                    f"{row['kf']:>12.4g}"
                    f"{row['kr']:>12.4g}"
                    f"{row['deltaG_kJ_per_mol']:>14.4g}"
                    f"{row['Keq']:>16.4g}"
                    f"{row['kf_over_kr']:>16.4g}"
                    f"{row['ratio_over_Keq']:>16.4g}"
                    f"{row['sabio_n']:>10d}"
                    f"{row['kf_source']:>12}"
                )

        # Optional TOML write-out
        if args.write_toml:
            if tomllib is None:
                # We will write TOML without reading; no dependency on tomllib for dumping
                pass
            out_path = Path(args.write_toml)
            lines = []
            lines.append("# Auto-generated glycolysis parameters")
            lines.append("")
            lines.append("[conditions]")
            lines.append(f"temperature = {T}")
            lines.append(f"pH = {args.ph}")
            lines.append(f"ionic_strength = {args.ionic_strength}")
            lines.append(f"pMg = {args.pmg}")
            lines.append("")
            lines.append("[reactions]")
            for step, formula in GLYCOLYSIS_REACTION_FORMULAS.items():
                lines.append(f"{step} = \"{formula}\"")
            lines.append("")
            lines.append("[deltaG]")
            for row in report:
                lines.append(f"{row['step']} = {row['deltaG_kJ_per_mol']}")
            lines.append("")
            lines.append("[deltaG_source]")
            for row in report:
                lines.append(f"{row['step']} = \"{row['deltaG_source']}\"")
            for i, row in enumerate(report, start=1):
                lines.append("")
                lines.append(f"[k{i}]")
                lines.append(f"kf = {row['kf']}")
                lines.append(f"kr = {row['kr']}")
                if row.get('EC'):
                    lines.append(f"ec = \"{row['EC']}\"")
            out_path.write_text("\n".join(lines), encoding="utf-8")
            print(f"\n✅ Wrote TOML to {out_path}")

        return 0
    except Exception as e:  # noqa: BLE001
        print(f"❌ KEGG processing failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
