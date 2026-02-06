# Tiger Stripe Matcher 🐯

A Streamlit app that compares stripe patterns across two tiger images.

**Pipeline**
- YOLOv8 ROI detection
- Segment Anything (SAM) segmentation
- Stripe enhancement + edge extraction
- Pattern matching + confidence score

## Run locally
```bash
pip install -r requirements.txt
streamlit run app/app.py
