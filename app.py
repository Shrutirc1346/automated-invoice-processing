#=============================
# Step 51.2 : Build First Live Application
#=============================
print("\n========== Step 51.2 : BUILD FIRST LIVE APPLICATION ========== ")

import os
import tempfile

import pandas as pd
import streamlit as st

from src.main import process_invoice
from src.anomaly_detector import detect_anomalies


#=============================
# Step 51.2 : Configure Streamlit Application
#=============================
print("\n========== Step 51.2 : CONFIGURE STREAMLIT APPLICATION ========== ")

st.set_page_config(
    page_title="Automated Invoice Processing",
    page_icon="🧾",
    layout="wide"
)

st.title(
    "🧾 Automated Invoice Processing and Anomaly Detection"
)

st.write(
    "Upload invoice images to extract invoice information, "
    "validate financial data, and detect potential anomalies."
)


#=============================
# Step 51.2 : Enable Multiple Image Upload
#=============================
print("\n========== Step 51.2 : ENABLE MULTIPLE IMAGE UPLOAD ========== ")

uploaded_files = st.file_uploader(
    "Upload Invoice Images",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    accept_multiple_files=True
)


#=============================
# Step 51.2 : Display Uploaded Images
#=============================
print("\n========== Step 51.2 : DISPLAY UPLOADED IMAGES ========== ")

if uploaded_files:

    st.subheader("Uploaded Invoices")

    for uploaded_file in uploaded_files:

        st.image(
            uploaded_file,
            caption=f"Uploaded Invoice: {uploaded_file.name}",
            use_container_width=True
        )


#=============================
# Step 51.2 : Process Uploaded Invoices
#=============================
print("\n========== Step 51.2 : PROCESS UPLOADED INVOICES ========== ")

if uploaded_files:

    if st.button("Process Invoices"):

        all_results = []

        progress_bar = st.progress(
            0
        )

        total_files = len(
            uploaded_files
        )

        for index, uploaded_file in enumerate(
            uploaded_files,
            start=1
        ):

            st.subheader(
                f"Processing: {uploaded_file.name}"
            )

            with tempfile.TemporaryDirectory() as temp_directory:

                image_path = os.path.join(
                    temp_directory,
                    uploaded_file.name
                )

                with open(
                    image_path,
                    "wb"
                ) as file:

                    file.write(
                        uploaded_file.getbuffer()
                    )

                with st.spinner(
                    f"Processing {uploaded_file.name}..."
                ):

                    try:

                        results = process_invoice(
                            image_path
                        )

                        if results:

                            all_results.extend(
                                results
                            )

                            st.success(
                                f"{uploaded_file.name} "
                                f"processed successfully."
                            )

                        else:

                            st.warning(
                                f"No invoice records extracted "
                                f"from {uploaded_file.name}."
                            )

                    except Exception as error:

                        st.error(
                            f"Processing failed for "
                            f"{uploaded_file.name}: {error}"
                        )

            progress_bar.progress(
                index / total_files
            )


        #=============================
        # Step 51.2 : Display Final Results
        #=============================
        print(
            "\n========== "
            "Step 51.2 : DISPLAY FINAL RESULTS "
            "========== "
        )

        if all_results:

        
            print("\n========== Step 51.2 : RUN ANOMALY DETECTION ========== ")

            results_dataframe = pd.DataFrame(
                all_results
            )

            anomaly_result = detect_anomalies(
                results_dataframe
            )

            all_results = anomaly_result.to_dict(
                orient="records"
            )

            st.success(
                f"{len(all_results)} invoice item record(s) "
                f"extracted successfully."
            )

            st.subheader(
                "Extracted Invoice Data"
            )

            st.dataframe(
                all_results,
                use_container_width=True
            )


            #=============================
            # Step 51.2 : Calculate Anomaly Summary
            #=============================
            print(
                "\n========== "
                "Step 51.2 : CALCULATE ANOMALY SUMMARY "
                "========== "
            )

            anomaly_records = [

                record

                for record in all_results

                if record.get(
                    "IsAnomaly"
                ) is True

            ]

            normal_records = [

                record

                for record in all_results

                if record.get(
                    "IsAnomaly"
                ) is False

            ]


            #=============================
            # Step 51.2 : Display Anomaly Summary
            #=============================
            print(
                "\n========== "
                "Step 51.2 : DISPLAY ANOMALY SUMMARY "
                "========== "
            )

            col1, col2, col3 = st.columns(
                3
            )

            with col1:

                st.metric(
                    "Total Records",
                    len(all_results)
                )

            with col2:

                st.metric(
                    "Normal Records",
                    len(normal_records)
                )

            with col3:

                st.metric(
                    "Potential Anomalies",
                    len(anomaly_records)
                )


            if anomaly_records:

                st.error(
                    f"⚠️ {len(anomaly_records)} "
                    f"potential anomalous record(s) detected."
                )

            else:

                st.success(
                    "✅ No potential anomalies detected."
                )


        else:

            st.warning(
                "No invoice records could be extracted."
            )


print(
    "\n========== STEP 51.2 FIRST LIVE VERSION READY ========== "
)