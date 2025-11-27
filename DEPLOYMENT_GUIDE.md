# 🚀 PythonAnywhere Deployment Guide

## Your Live Website Link
After completing this guide, your website will be live at:
```
https://salomi-p.pythonanywhere.com
```

## Prerequisites Ready ✅
- ✅ Procfile created with deployment command
- ✅ gunicorn==21.2.0 added to requirements.txt
- ✅ All files committed to GitHub

## Step 1: Sign Up (FREE - No Credit Card Required)
1. Go to https://www.pythonanywhere.com
2. Click **"Pricing & signup"** in the top right
3. Choose **"Create a free account"**
4. Fill in your details:
   - Username: `salomi-p`
   - Email: Your email
   - Password: Your secure password
5. Click "I agree to the Terms of Service"
6. Sign up and verify your email

## Step 2: Create Web App in Dashboard
1. Log in to PythonAnywhere Dashboard
2. Click **"Web"** tab on the left
3. Click **"+ Add a new web app"**
4. Choose configuration:
   - **Python version**: Python 3.10
   - **Web framework**: Flask

## Step 3: Clone Your Repository
1. Click **"Consoles"** tab
2. Click **"Bash"** console
3. Run these commands:
```bash
cd /home/salomi-p
git clone https://github.com/salomi-p/A-Lexicon-Enhanced-LSTM-Framework-for-Sentiment-Aware-Fake-News-Detection.git
cd A-Lexicon-Enhanced-LSTM-Framework-for-Sentiment-Aware-Fake-News-Detection
```

## Step 4: Create Virtual Environment
In the Bash console, run:
```bash
mkvirtualenv --python=/usr/bin/python3.10 myenv
source /home/salomi-p/.virtualenvs/myenv/bin/activate
pip install -r requirements.txt
```

## Step 5: Configure WSGI
1. Go back to **"Web"** tab
2. Click your web app name
3. Under "Code" section, click on **"WSGI configuration file"**
4. Delete all content and replace with:
```python
import sys
path = '/home/salomi-p/A-Lexicon-Enhanced-LSTM-Framework-for-Sentiment-Aware-Fake-News-Detection'
if path not in sys.path:
    sys.path.append(path)
from main import app as application
```
5. Click "Save"

## Step 6: Set Virtual Environment Path
Still in the "Web" tab:
1. Under "Virtualenv" section, enter:
```
/home/salomi-p/.virtualenvs/myenv
```
2. Click outside to save

## Step 7: Go Live!
1. Click the green **"Reload"** button at the top
2. Wait 10-15 seconds for the app to restart
3. Visit: **`https://salomi-p.pythonanywhere.com`** ✅

Your Flask app is now LIVE!

## Troubleshooting

### App shows error after reload
- Check the "Error logs" tab in Web settings
- Verify the WSGI configuration file is correct
- Make sure requirements.txt installed successfully

### Can't clone repository
- Verify internet connection
- Check GitHub URL is correct
- Try: `git clone` with full URL

### Database connection issues
- Update MySQL connection strings in main.py
- For local development, you may need to set up MySQL in PythonAnywhere
- Or modify app to use SQLite database

## Database Configuration

If you need to use MySQL in PythonAnywhere:
1. Go to "Databases" tab
2. Create a new MySQL database
3. Note the database details
4. Update main.py:
```python
mydb = mysql.connector.connect(
    host="salomi-p.mysql.pythonanywhere-services.com",
    user="salomi-p",
    password="your_password",
    database="salomi-p$fake_news_classify"
)
```

## Live Website Details

- **URL**: https://salomi-p.pythonanywhere.com
- **Free Tier Features**:
  - 512 MB disk space
  - Python web apps with Flask/Django
  - One web app running
  - Basic MySQL database
  - Free SSL certificate (HTTPS)

## Next Steps (Optional)

1. **Upgrade Plan**: If you need more power, upgrade to a paid plan
2. **Custom Domain**: Add your own domain name
3. **Email Notifications**: Configure alerts for uptime
4. **Backups**: Enable automatic database backups

---

**Your Flask app is deployment-ready! Follow the steps above to deploy.** 🎉
