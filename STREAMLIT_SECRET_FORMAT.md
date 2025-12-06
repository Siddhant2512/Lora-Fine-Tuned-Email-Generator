# Streamlit Cloud Secret Format - IMPORTANT

## Correct Format for Adding HF_TOKEN

When adding secrets in Streamlit Cloud, you need to use **TOML format**, not just KEY/VALUE pairs.

### Step-by-Step Instructions:

1. **Go to Streamlit Cloud**: https://share.streamlit.io/
2. **Select your app**
3. **Click "Settings"** (⚙️ icon) or **"Manage app"**
4. **Click "Secrets"** tab
5. **In the secrets editor**, you should see a text box. Enter this **EXACT format**:

```toml
HF_TOKEN = "your_huggingface_token_here"
```

**Important points:**
- ✅ Use `=` (equals sign), not `:`
- ✅ Put the value in **double quotes** `"..."`  
- ✅ No spaces around the `=` (or one space on each side is fine)
- ✅ The key `HF_TOKEN` should be uppercase
- ✅ Make sure there are no extra characters or spaces

### ❌ WRONG Formats (Don't use these):

```toml
# Wrong - no quotes
HF_TOKEN = your_huggingface_token_here

# Wrong - using colon
HF_TOKEN: "your_huggingface_token_here"

# Wrong - extra spaces or formatting
HF_TOKEN  =  "your_huggingface_token_here"
```

### ✅ CORRECT Format:

```toml
HF_TOKEN = "your_huggingface_token_here"
```

## After Adding the Secret:

1. **Click "Save"**
2. **Wait for automatic redeploy** (or manually trigger redeploy)
3. **Check the debug section** in the app sidebar (expand "🔍 Debug Info")
4. It should show: ✅ HF_TOKEN found in secrets

## Troubleshooting:

If it still doesn't work:

1. **Check the debug section** in the app to see if the token is being read
2. **Verify the format** matches exactly what's shown above
3. **Make sure you saved** the secret (not just typed it)
4. **Redeploy** the app after saving
5. **Check Streamlit Cloud logs** for any errors

## Your Token:

Use your actual Hugging Face token (starts with `hf_`). Make sure to include the quotes when adding to Streamlit Cloud!

