from pathlib import Path
from datetime import datetime

path = Path("app_v2.py")
source = path.read_text(encoding="utf-8")

start_marker = "                headers = list(rows[0].keys()) if rows else []"
end_marker = "                    story.extend([report_table, Spacer(1, 5 * mm)])"

start = source.find(start_marker)
if start < 0:
    raise SystemExit("ERROR: Table-generation start marker not found.")

end = source.find(end_marker, start)
if end < 0:
    raise SystemExit("ERROR: Table-generation end marker not found.")

end += len(end_marker)

replacement = '''                headers = list(rows[0].keys()) if rows else []
                table_data = []

                # Landscape A4 printable width: 297 mm minus 30 mm margins.
                available_width = landscape(A4)[0] - 30 * mm

                from reportlab.lib.styles import ParagraphStyle
                from reportlab.platypus import KeepTogether

                cell_style = ParagraphStyle(
                    "MetricCell",
                    parent=styles["BodyText"],
                    fontName="Helvetica",
                    fontSize=6.5,
                    leading=8,
                    spaceAfter=0,
                )
                header_style = ParagraphStyle(
                    "MetricHeader",
                    parent=cell_style,
                    fontName="Helvetica-Bold",
                    fontSize=6,
                    leading=7,
                    textColor=colors.white,
                )

                # Shorten visual labels while preserving the full CSV column names.
                display_headers = {
                    "Class": "Class",
                    "Ground-truth voxels": "GT voxels",
                    "Baseline voxels": "Baseline voxels",
                    "Hybrid voxels": "Hybrid voxels",
                    "Baseline Dice": "Baseline Dice",
                    "Hybrid Dice": "Hybrid Dice",
                    "Ground-truth volume (mL)": "GT volume (mL)",
                    "Baseline volume (mL)": "Baseline volume (mL)",
                    "Hybrid volume (mL)": "Hybrid volume (mL)",
                    "Baseline absolute volume difference (mL)": "Baseline abs. diff. (mL)",
                    "Hybrid absolute volume difference (mL)": "Hybrid abs. diff. (mL)",
                }

                if headers:
                    table_data.append([
                        Paragraph(
                            display_headers.get(h, h).replace("&", "&amp;"),
                            header_style,
                        )
                        for h in headers
                    ])

                    for row in rows:
                        table_data.append([
                            Paragraph(
                                str(row.get(h, "")).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"),
                                cell_style,
                            )
                            for h in headers
                        ])

                if table_data and headers:
                    # Allocate widths based on content type, then scale to page width.
                    weights = []
                    for header in headers:
                        name = header.lower()
                        if "class" in name:
                            weights.append(0.75)
                        elif "dice" in name:
                            weights.append(0.85)
                        elif "voxels" in name:
                            weights.append(1.05)
                        elif "volume" in name or "difference" in name:
                            weights.append(1.25)
                        else:
                            weights.append(1.0)

                    total_weight = sum(weights)
                    col_widths = [
                        available_width * weight / total_weight
                        for weight in weights
                    ]

                    report_table = Table(
                        table_data,
                        colWidths=col_widths,
                        repeatRows=1,
                        hAlign="LEFT",
                    )
                    report_table.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#23395B")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                         [colors.white, colors.HexColor("#EEF2F7")]),
                        ("LEFTPADDING", (0, 0), (-1, -1), 3),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]))
                    story.extend([report_table, Spacer(1, 5 * mm)])'''

backup = path.with_name(
    f"app_v2_before_pdf_table_fix_{datetime.now():%Y%m%d_%H%M%S}.py"
)
backup.write_text(source, encoding="utf-8")
updated = source[:start] + replacement + source[end:]
path.write_text(updated, encoding="utf-8")

print(f"Backup created: {backup}")
print("PDF table layout updated.")
