# Streamlit Cloud Setup Guide

## Setting Up Hugging Face Token for Llama Models

To use Llama 3.2 3B on Streamlit Cloud, you need to configure your Hugging Face token as a secret.

### Step 1: Get Your Hugging Face Token

1. Go to: **https://huggingface.co/settings/tokens**
2. Click **"New token"**
3. **Token name**: `lora-mail-assistant` (or any name)
4. **Token type**: Select **"Read"** (this is enough for downloading models)
5. Click **"Generate token"**
6. **IMPORTANT**: Copy the token immediately (you won't see it again!)

### Step 2: Request Access to Llama Model

1. Go to: **https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct**
2. Click **"Request access"** button
3. Wait for approval (usually instant or within a few hours)
4. Once approved, you can use the model

### Step 3: Add Token to Streamlit Cloud

1. Go to your Streamlit Cloud dashboard: **https://share.streamlit.io/**
2. Select your app
3. Click **"Settings"** (⚙️ icon) or **"Manage app"**
4. Click on **"Secrets"** tab
5. Click **"Add new secret"** or edit the secrets file
6. Add the following:

```toml
HF_TOKEN = "your_huggingface_token_here"
```

**Important**: Replace `your_huggingface_token_here` with your actual token from Step 1.

7. Click **"Save"**

### Step 4: Redeploy Your App

1. After saving the secret, Streamlit Cloud will automatically redeploy
2. Or manually trigger a redeploy from the app settings
3. Wait for deployment to complete

### Step 5: Verify It Works

1. Open your deployed app
2. Try generating an email
3. The model should now load without authentication errors

## Alternative: Using Environment Variables

If you're deploying elsewhere (not Streamlit Cloud), set the environment variable:

```bash
export HF_TOKEN=your_huggingface_token_here
```

Or in your deployment platform's environment variables section, add:
- **Key**: `HF_TOKEN`
- **Value**: Your Hugging Face token

## Troubleshooting

### Error: "Authentication required"
- ✅ Make sure you've added `HF_TOKEN` to Streamlit Cloud secrets
- ✅ Verify the token is correct (no extra spaces)
- ✅ Make sure you have access to the Llama model
- ✅ Redeploy the app after adding the secret

### Error: "Model not found" or "401 Unauthorized"
- ✅ Request access to the model: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
- ✅ Wait for approval
- ✅ Make sure your token has "Read" permissions

### Token Not Working
- ✅ Regenerate a new token if needed
- ✅ Make sure token type is "Read" (not "Write")
- ✅ Check that token hasn't expired (they don't expire, but check anyway)

## Security Notes

- ✅ **Never commit your token to Git** - it's already in `.gitignore`
- ✅ **Use Streamlit Secrets** - they're encrypted and secure
- ✅ **Read token is enough** - you don't need Write permissions
- ✅ **Token is per-app** - each Streamlit app has its own secrets

## Quick Checklist

- [ ] Created Hugging Face account
- [ ] Generated Read token
- [ ] Requested access to Llama 3.2 3B model
- [ ] Added `HF_TOKEN` to Streamlit Cloud secrets
- [ ] Redeployed the app
- [ ] Verified it works

## Need Help?

If you're still having issues:
1. Check Streamlit Cloud logs for detailed error messages
2. Verify token is set correctly in secrets
3. Make sure model access is approved
4. Try regenerating the token

