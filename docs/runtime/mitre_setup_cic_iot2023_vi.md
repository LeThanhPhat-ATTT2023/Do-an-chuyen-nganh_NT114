# Huong dan lay khung MITRE cho CIC-IoT2023

> ⚠️ CAP NHAT (v3_ob, 2026-06): Buoc lay ATT&CK CSV/STIX van dung. Nhung
> "tactical edges packet/flow -> technique tao boi cosine similarity" (muc duoi)
> da bi thay the. v3_ob khong dung cosine giua embedding student va technique;
> thay bang MSEE (PMI counting + L1 LR + Aho-Corasick procedure matcher tren STIX
> enterprise-attack.json). Edge mang family/weight/source/provenance. Nguon chuan:
> CLAUDE.md + docs/architecture/system_execution_flows.md.

## Muc tieu
Lay bo ATT&CK chinh thong va tao cac bang node/edge de dung cho tang ngu nghia trong do thi 3 tang.

## Nguon chuan
1. ATT&CK Data & Tools: https://attack.mitre.org/resources/attack-data-and-tools/
2. STIX 2.0 (MITRE CTI): https://github.com/mitre/cti

## Buoc 1: Tai du lieu STIX enterprise ATT&CK

```powershell
$outDir = "data/mitre"
New-Item -ItemType Directory -Path $outDir -Force | Out-Null
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json" -OutFile "$outDir/enterprise-attack.json"
```

## Buoc 2: Trich xuat bang node/edge MITRE

```powershell
graphslm-prepare-mitre --input-json "data/mitre/enterprise-attack.json"
```

Du lieu tao ra:
1. data/mitre/mitre_techniques.csv
2. data/mitre/mitre_tactics.csv
3. data/mitre/mitre_technique_tactic_edges.csv
4. data/mitre/mitre_export_stats.json

## Buoc 3: Seed mapping cho CIC-IoT2023

Seed file da co san o:
1. configs/cic_iot2023_to_mitre_seed.csv

Luu y:
1. Day la mapping khoi dau de bootstrapping.
2. Can review thu cong theo nhan con cua CIC-IoT2023 de nang do chinh xac.

## Buoc 4: Dung cho tang 3 cua do thi

1. Technique nodes: doc tu data/mitre/mitre_techniques.csv
2. Technique-tactic edges: doc tu data/mitre/mitre_technique_tactic_edges.csv
3. Tactical edges packet/flow -> technique: tao boi cosine similarity giua embedding student va node technique.

## Ghi chu thuc nghiem
1. Khi mapping MITRE cho online path, nen luu top-k technique va score.
2. Nen dat nguong ban dau 0.85 va tune bang ablation tren CIC-IoT2023.
