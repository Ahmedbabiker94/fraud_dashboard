# Uploading this release to GitHub

The connected GitHub integration returned **403 Resource not accessible by integration** when creating a branch. These release files have therefore **not been pushed**.

To update from a local computer or Google Colab with write access:

1. Download and extract `fraud_portfolio_release.zip`.
2. Clone `https://github.com/Ahmedbabiker94/fraud_dashboard.git` to your computer.
3. Run `git checkout -b portfolio/checkpoint-dashboard-report`.
4. Copy all files from the extracted release folder into the repository, overwriting `dashboard.py` and `requirements.txt`. Do **not** copy `venv/` or your full `creditcard.csv` into the repository.
5. Run `pip install -r requirements.txt`, `pytest -q`, and `streamlit run dashboard.py` in a working Python environment with `torch-geometric` installed.
6. Validate model inference on several batches and compare with Colab evaluation before publishing an app link.
7. Commit and push using `git add README.md dashboard.py hybrid_model.py model_inference.py requirements.txt .gitignore docs tests model/fraud_model.pt data/creditcard_sample.csv HybridGNNVAEGRU_RAM_Fixed.ipynb`, `git commit -m "Document experiment and align dashboard with original checkpoint"`, and `git push -u origin portfolio/checkpoint-dashboard-report`.
8. Open a pull request into `main`. Review before merging.

**Do not push the full creditcard.csv or credentials.** Model/checkpoint files should be inspected and loaded using `torch.load(weights_only=True)` wherever possible. The old `model/fraud_model.pkl` should be removed after the new dashboard is deployed and checked.

The project does **not** have independently reproduced model metrics in this release. A full test of the dashboard in a supported PyTorch Geometric + Streamlit environment is still required.
