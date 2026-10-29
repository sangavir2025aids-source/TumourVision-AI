import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image
)


def generate_pdf_report(
    patient_id="",
    name="",
    age="",
    email="",
    filename="",
    result="",
    tumour_pixels=0,
    tumour_size="",
    small_count=0,
    medium_count=0,
    large_count=0,
    detected_regions=0,
    approximate_location="",
    highlighted_image=""
):

    # --------------------------------------------------
    # OUTPUT FOLDER
    # --------------------------------------------------

    output_folder = "outputs"

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    # --------------------------------------------------
    # PATIENT ID FOR FILE NAME
    # --------------------------------------------------

    safe_patient_id = patient_id.strip() if patient_id else "Unknown"

    # Remove characters that are unsafe for filenames
    safe_patient_id = "".join(
        c for c in safe_patient_id
        if c.isalnum() or c in ("-", "_")
    )

    if not safe_patient_id:
        safe_patient_id = "Unknown"

    pdf_filename = (
        f"TumourVision_Report_{safe_patient_id}.pdf"
    )

    pdf_path = os.path.join(
        output_folder,
        pdf_filename
    )

    # --------------------------------------------------
    # DOCUMENT
    # --------------------------------------------------

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontSize=24,
        leading=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#064E3B"),
        spaceAfter=10
    )

    subtitle_style = ParagraphStyle(
        "SubtitleStyle",
        parent=styles["Normal"],
        fontSize=11,
        leading=16,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#315477"),
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading2"],
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#064E3B"),
        spaceBefore=12,
        spaceAfter=10
    )

    normal_style = ParagraphStyle(
        "NormalStyle",
        parent=styles["Normal"],
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#172554")
    )

    disclaimer_style = ParagraphStyle(
        "DisclaimerStyle",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=13,
        textColor=colors.HexColor("#555555")
    )

    # --------------------------------------------------
    # STORY
    # --------------------------------------------------

    story = []

    # --------------------------------------------------
    # HEADER
    # --------------------------------------------------

    story.append(
        Paragraph(
            "TumourVision AI",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Precision Liver Tumour Segmentation using AI",
            subtitle_style
        )
    )

    story.append(
        Paragraph(
            "AI-Assisted CT Scan Analysis Report",
            subtitle_style
        )
    )

    story.append(
        Spacer(1, 10)
    )

    # --------------------------------------------------
    # PATIENT INFORMATION
    # --------------------------------------------------

    story.append(
        Paragraph(
            "Patient Information",
            heading_style
        )
    )

    patient_data = [
        ["Patient ID", patient_id or "Not provided"],
        ["Patient Name", name or "Not provided"],
        ["Age", age or "Not provided"],
        ["Email", email or "Not provided"],
        ["CT Scan", filename or "Not provided"],
        [
            "Analysis Date",
            datetime.now().strftime("%d-%m-%Y %H:%M")
        ]
    ]

    patient_table = Table(
        patient_data,
        colWidths=[1.6 * inch, 4.8 * inch]
    )

    patient_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor("#D1FAE5")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, -1),
                colors.HexColor("#172554")
            ),
            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),
            (
                "FONTNAME",
                (1, 0),
                (1, -1),
                "Helvetica"
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#A7F3D0")
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                8
            )
        ])
    )

    story.append(patient_table)

    story.append(
        Spacer(1, 20)
    )

    # --------------------------------------------------
    # AI ANALYSIS
    # --------------------------------------------------

    story.append(
        Paragraph(
            "AI Analysis Results",
            heading_style
        )
    )

    detection_status = (
        "Tumour Detected"
        if int(tumour_pixels or 0) > 0
        else "No Tumour Detected"
    )

    analysis_data = [
        ["Detection Status", detection_status],
        ["Total Tumour Area", f"{tumour_pixels} pixels"],
        ["Small Tumours", str(small_count)],
        ["Medium Tumours", str(medium_count)],
        ["Large Tumours", str(large_count)],
        ["Detected Regions", str(detected_regions)],
        ["Overall Size", tumour_size or "Not available"],
        [
            "Tumour Location",
            approximate_location or "Not available"
        ]
    ]

    analysis_table = Table(
        analysis_data,
        colWidths=[2.2 * inch, 4.2 * inch]
    )

    analysis_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor("#D1FAE5")
            ),
            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),
            (
                "FONTNAME",
                (1, 0),
                (1, -1),
                "Helvetica"
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, -1),
                colors.HexColor("#172554")
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#A7F3D0")
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                8
            )
        ])
    )

    story.append(analysis_table)

    story.append(
        Spacer(1, 20)
    )

    # --------------------------------------------------
    # AI SEGMENTATION IMAGE
    # --------------------------------------------------

    story.append(
        Paragraph(
            "AI Tumour Segmentation",
            heading_style
        )
    )

    # The app currently sends only the filename.
    # Convert it into the actual outputs path.

    if highlighted_image:

        if os.path.exists(highlighted_image):

            image_path = highlighted_image

        else:

            image_path = os.path.join(
                output_folder,
                highlighted_image
            )

        if os.path.exists(image_path):

            try:

                img = Image(
                    image_path,
                    width=6.5 * inch,
                    height=4.8 * inch
                )

                story.append(img)

                story.append(
                    Spacer(1, 10)
                )

                story.append(
                    Paragraph(
                        "Green = Small Tumour | "
                        "Orange = Medium Tumour | "
                        "Red = Large Tumour",
                        normal_style
                    )
                )

            except Exception:

                story.append(
                    Paragraph(
                        "AI segmentation image could not be added.",
                        normal_style
                    )
                )

        else:

            story.append(
                Paragraph(
                    "AI segmentation image was not found.",
                    normal_style
                )
            )

    else:

        story.append(
            Paragraph(
                "AI segmentation image was not provided.",
                normal_style
            )
        )

    story.append(
        Spacer(1, 20)
    )


    # --------------------------------------------------
    # BUILD PDF
    # --------------------------------------------------

    doc.build(story)

    print(
        f"PDF report generated successfully: {pdf_path}"
    )

    return pdf_path