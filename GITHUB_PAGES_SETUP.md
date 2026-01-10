# GitHub Pages Setup Instructions

## Quick Setup

1. Go to your repository: https://github.com/mahdiarsadeghi/gold

2. Click on **Settings** (top right)

3. In the left sidebar, click **Pages**

4. Under "Build and deployment":
   - Source: **Deploy from a branch**
   - Branch: **main**
   - Folder: **/docs**
   - Click **Save**

5. Wait 2-3 minutes for GitHub to build and deploy

6. Your site will be live at: **https://mahdiarsadeghi.github.io/gold/**

## Verify Deployment

- Go to the **Actions** tab in your repository
- You should see a "pages build and deployment" workflow running
- Once completed (green checkmark), your site is live!

## Automated Updates

The GitHub Actions workflow will automatically:
- Run predictions every Monday at 9 AM UTC
- Update the web dashboard with new predictions
- Commit and push the changes

You can also manually trigger a prediction:
1. Go to **Actions** tab
2. Click "Weekly Gold Price Prediction"
3. Click "Run workflow"

## Local Testing

To test the website locally before deployment:

```bash
cd docs
python -m http.server 8000
```

Then open: http://localhost:8000

## Troubleshooting

If the page doesn't load:
1. Check GitHub Pages settings are correct
2. Ensure the docs folder is pushed to main branch
3. Run the predictor at least once to generate data.json
4. Check the Actions tab for any errors
