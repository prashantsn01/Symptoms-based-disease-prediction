# How to Push This to Your Own GitHub

Follow these steps to deploy this project from scratch on your GitHub account.

---

## Step 1 – Create a new repo on GitHub

1. Go to https://github.com/new
2. Name it: `symptoms-based-disease-prediction`
3. Set it to **Public** or **Private** (your choice)
4. ❌ Do NOT initialise with README, .gitignore, or license (we already have them)
5. Click **Create repository**

---

## Step 2 – Initialise git locally

Open a terminal in this project folder and run:

```bash
git init
git add .
git commit -m "Initial commit: disease prediction system"
```

---

## Step 3 – Connect to your GitHub repo

Replace `<your-username>` with your actual GitHub username:

```bash
git remote add origin https://github.com/<your-username>/symptoms-based-disease-prediction.git
git branch -M main
git push -u origin main
```

---

## Step 4 – Verify

Visit `https://github.com/<your-username>/symptoms-based-disease-prediction`  
You should see all your files and the README rendered on the page.

---

## Notes

- The `data/` and `models/` folders are **excluded from git** (they contain large files).  
  Anyone cloning your repo will need to add the dataset and run training themselves.
- If you have a `GEMINI_API_KEY`, never commit it — use a `.env` file (already in .gitignore).

---

## Optional: Add a GitHub Actions badge

After your first push, you can add CI by creating `.github/workflows/lint.yml`.
