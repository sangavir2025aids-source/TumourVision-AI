from flask import (
    Flask,
    render_template,
    request,
    session,
    send_from_directory,
    redirect,
    send_file
)

import os
from datetime import datetime
from werkzeug.utils import secure_filename

from predict import predict_tumour
from pdf_report import generate_pdf_report


app = Flask(__name__)

app.secret_key = "tumourvision-secret-key"


# =====================================================
# FOLDERS
# =====================================================

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# =====================================================
# HOME
# =====================================================

@app.route("/")
def home():

    return redirect("/dashboard")


# =====================================================
# DASHBOARD
# =====================================================

@app.route("/dashboard")
def dashboard():

    return render_template(
        "dashboard.html"
    )


# =====================================================
# PATIENT ANALYSIS
# =====================================================

@app.route(
    "/patient-analysis",
    methods=["GET", "POST"]
)
def patient_analysis():

    saved = False

    if request.method == "POST":

        # Save only basic patient information

        session["patient_id"] = (
            request.form.get("patient_id")
        )

        session["name"] = (
            request.form.get("name")
        )

        session["age"] = (
            request.form.get("age")
        )

        session["email"] = (
            request.form.get("email")
        )

        saved = True

    return render_template(

        "patient_analysis.html",

        patient_id=session.get(
            "patient_id",
            ""
        ),

        name=session.get(
            "name",
            ""
        ),

        age=session.get(
            "age",
            ""
        ),

        email=session.get(
            "email",
            ""
        ),

        saved=saved
    )


# =====================================================
# CT SCAN
# =====================================================

@app.route("/ct-scan")
def ct_scan():

    return render_template(

        "ct_scan.html",

        patient_id=session.get(
            "patient_id",
            ""
        ),

        name=session.get(
            "name",
            ""
        ),

        age=session.get(
            "age",
            ""
        ),

        email=session.get(
            "email",
            ""
        ),

        filename=session.get(
            "filename",
            ""
        ),

        result=session.get(
            "result",
            ""
        ),

        tumour_pixels=session.get(
            "tumour_pixels",
            0
        ),

        tumour_size=session.get(
            "tumour_size",
            ""
        ),

        highlighted_filename=session.get(
            "highlighted_filename",
            ""
        ),

        small_count=session.get(
            "small_count",
            0
        ),

        medium_count=session.get(
            "medium_count",
            0
        ),

        large_count=session.get(
            "large_count",
            0
        ),

        detected_regions=session.get(
            "detected_regions",
            0
        ),

        approximate_location=session.get(
            "approximate_location",
            ""
        )
    )


# =====================================================
# AI RESULTS
# =====================================================

@app.route("/ai-results")
def ai_results():

    return render_template(

        "ai_results.html",

        patient_id=session.get(
            "patient_id",
            ""
        ),

        name=session.get(
            "name",
            ""
        ),

        age=session.get(
            "age",
            ""
        ),

        email=session.get(
            "email",
            ""
        ),

        filename=session.get(
            "filename",
            ""
        ),

        result=session.get(
            "result",
            ""
        ),

        tumour_pixels=session.get(
            "tumour_pixels",
            0
        ),

        tumour_size=session.get(
            "tumour_size",
            ""
        ),

        highlighted_filename=session.get(
            "highlighted_filename",
            ""
        ),

        small_count=session.get(
            "small_count",
            0
        ),

        medium_count=session.get(
            "medium_count",
            0
        ),

        large_count=session.get(
            "large_count",
            0
        ),

        detected_regions=session.get(
            "detected_regions",
            0
        ),

        approximate_location=session.get(
            "approximate_location",
            ""
        )
    )


# =====================================================
# REPORTS
# =====================================================

@app.route("/reports")
def reports():

    return render_template(
        "reports.html"
    )


# =====================================================
# UPLOAD CT SCAN
# =====================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload():

    if "ct_scan" not in request.files:

        return "No CT scan selected."


    file = request.files["ct_scan"]


    if file.filename == "":

        return "No CT scan selected."


    filename = secure_filename(
        file.filename
    )


    file_path = os.path.join(

        app.config["UPLOAD_FOLDER"],

        filename
    )


    file.save(file_path)


    try:

        (
            original,
            predicted_mask,
            tumour_pixels,
            highlighted_path,
            small_count,
            medium_count,
            large_count,
            detected_regions,
            approximate_location

        ) = predict_tumour(
            file_path
        )


        # ==========================================
        # TUMOUR STATUS
        # ==========================================

        if tumour_pixels > 0:

            result = "Tumour Detected"


            if tumour_pixels < 500:

                tumour_size = "Small"


            elif tumour_pixels <= 2000:

                tumour_size = "Medium"


            else:

                tumour_size = "Large"


        else:

            result = "No Tumour Detected"

            tumour_size = "None"


        highlighted_filename = os.path.basename(
            highlighted_path
        )


    except Exception as e:

        print(
            "Prediction error:",
            e
        )


        result = "Prediction Error"

        tumour_pixels = 0

        tumour_size = "Unknown"

        highlighted_filename = None

        small_count = 0

        medium_count = 0

        large_count = 0

        detected_regions = 0

        approximate_location = "Not available"


    # ==========================================
    # SAVE AI RESULTS IN SESSION
    # ==========================================

    session["filename"] = filename

    session["result"] = result

    session["tumour_pixels"] = tumour_pixels

    session["tumour_size"] = tumour_size

    session["highlighted_filename"] = (
        highlighted_filename
    )

    session["small_count"] = small_count

    session["medium_count"] = medium_count

    session["large_count"] = large_count

    session["detected_regions"] = (
        detected_regions
    )

    session["approximate_location"] = (
        approximate_location
    )


    # After AI analysis,
    # go automatically to AI Results

    return redirect(
        "/ai-results"
    )


# =====================================================
# UPLOADED FILES
# =====================================================

@app.route(
    "/uploads/<filename>"
)
def uploaded_file(filename):

    return send_from_directory(

        UPLOAD_FOLDER,

        filename
    )


# =====================================================
# OUTPUT FILES
# =====================================================

@app.route(
    "/outputs/<filename>"
)
def output_file(filename):

    return send_from_directory(

        OUTPUT_FOLDER,

        filename
    )


# =====================================================
# GENERATE PDF REPORT
# =====================================================

@app.route(
    "/generate-report"
)
def generate_report():

    try:

        pdf_path = generate_pdf_report(

            patient_id=session.get(
                "patient_id",
                ""
            ),

            name=session.get(
                "name",
                ""
            ),

            age=session.get(
                "age",
                ""
            ),

            email=session.get(
                "email",
                ""
            ),

            filename=session.get(
                "filename",
                ""
            ),

            result=session.get(
                "result",
                ""
            ),

            tumour_pixels=session.get(
                "tumour_pixels",
                0
            ),

            tumour_size=session.get(
                "tumour_size",
                ""
            ),

            small_count=session.get(
                "small_count",
                0
            ),

            medium_count=session.get(
                "medium_count",
                0
            ),

            large_count=session.get(
                "large_count",
                0
            ),

            detected_regions=session.get(
                "detected_regions",
                0
            ),

            highlighted_image=session.get(
                "highlighted_filename",
                ""
            ),

            # Current AI location
            approximate_location=session.get(
                "approximate_location",
                ""
            )
        )


        return send_file(

            pdf_path,

            as_attachment=True,

            download_name=os.path.basename(
                pdf_path
            )
        )


    except Exception as e:

        print(
            "PDF generation error:",
            e
        )

        return (
            f"PDF generation failed: {e}"
        )


# =====================================================
# RUN APPLICATION
# =====================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )