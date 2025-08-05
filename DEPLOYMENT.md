# Wage Prediction App Deployment Guide

This guide explains how to deploy the Wage Prediction application to Vercel (frontend) and various backend hosting options.

## Architecture Overview

The application consists of:
- **Frontend**: React app with TypeScript
- **Backend**: FastAPI server with machine learning capabilities

## Frontend Deployment (Vercel)

### Prerequisites
1. Vercel account (https://vercel.com)
2. Git repository (GitHub, GitLab, or Bitbucket)

### Deployment Steps

1. **Push your code to Git**
   ```bash
   git add .
   git commit -m "Prepare for Vercel deployment"
   git push origin main
   ```

2. **Deploy to Vercel**
   - Go to https://vercel.com/new
   - Import your repository
   - Select the `frontend` directory as the root directory
   - Vercel will auto-detect the Create React App framework
   - Configure environment variables:
     - `REACT_APP_API_URL`: Your backend API URL (e.g., `https://your-backend-api.com`)

3. **Alternative: Deploy via CLI**
   ```bash
   cd frontend
   npm install -g vercel
   vercel
   ```

## Backend Deployment Options

Since the backend uses FastAPI with ML models and file storage, you'll need a platform that supports Python applications and persistent storage.

### Option 1: Railway (Recommended)

Railway supports Python apps and provides persistent storage.

1. **Create account** at https://railway.app
2. **Create new project** and connect your GitHub repo
3. **Configure service**:
   ```toml
   # railway.toml in backend directory
   [build]
   builder = "NIXPACKS"
   
   [deploy]
   startCommand = "uvicorn app.main:app --host 0.0.0.0 --port $PORT"
   ```
4. **Add environment variables** in Railway dashboard
5. **Deploy** - Railway will auto-deploy on git push

### Option 2: Render

1. **Create account** at https://render.com
2. **Create Web Service**
3. **Configure**:
   - Runtime: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. **Add persistent disk** for file storage
5. **Deploy**

### Option 3: Google Cloud Run

For a more scalable solution with managed ML capabilities:

1. **Create Dockerfile** in backend directory:
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
   ```

2. **Deploy to Cloud Run**:
   ```bash
   gcloud run deploy wage-prediction-api \
     --source . \
     --region us-central1 \
     --allow-unauthenticated
   ```

### Option 4: AWS Elastic Beanstalk

1. **Install EB CLI**: `pip install awsebcli`
2. **Initialize**: `eb init -p python-3.9 wage-prediction-backend`
3. **Create environment**: `eb create wage-prediction-env`
4. **Deploy**: `eb deploy`

### Option 5: Heroku

1. **Create `Procfile`** in backend directory:
   ```
   web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
   ```

2. **Deploy**:
   ```bash
   heroku create wage-prediction-api
   git push heroku main
   ```

## Important Configuration Updates

### 1. Update CORS Settings

In `backend/app/main.py`, update CORS origins for production:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-vercel-app.vercel.app",
        "https://your-custom-domain.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 2. Environment Variables

Create `.env` file for backend:
```
DATABASE_URL=your_database_url
SECRET_KEY=your_secret_key
ENVIRONMENT=production
```

### 3. File Storage

For production, consider using cloud storage instead of local files:
- AWS S3
- Google Cloud Storage
- Cloudinary (for images)

## Post-Deployment Checklist

- [ ] Frontend deployed and accessible
- [ ] Backend API deployed and responding
- [ ] Environment variables configured correctly
- [ ] CORS settings updated for production URLs
- [ ] SSL certificates active (HTTPS)
- [ ] File upload functionality tested
- [ ] ML model predictions working
- [ ] Error monitoring set up (Sentry, etc.)

## Monitoring and Maintenance

1. **Set up monitoring**:
   - Vercel Analytics (frontend)
   - Backend API monitoring (UptimeRobot, Pingdom)
   - Error tracking (Sentry)

2. **Regular updates**:
   - Keep dependencies updated
   - Monitor ML model performance
   - Regular backups of uploaded data

## Troubleshooting

### Frontend Issues
- Check browser console for errors
- Verify API URL environment variable
- Check CORS errors

### Backend Issues
- Check application logs
- Verify file permissions for uploads
- Ensure ML models are included in deployment
- Check memory limits for ML operations