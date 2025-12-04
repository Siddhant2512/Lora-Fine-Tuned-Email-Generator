# Hugging Face Token Setup Guide

## Step-by-Step Token Creation

### Step 1: Navigate to Token Settings

1. Go to: **https://huggingface.co/settings/tokens**
2. You'll need to be logged into your Hugging Face account
3. If not logged in, sign up/login at: https://huggingface.co/join

### Step 2: Create New Token

Click the **"New token"** button (usually top-right or in the tokens list)

### Step 3: Configure Token Details

#### Token Name
- **What to enter**: `lora-mail-assistant` or `email-assistant` or any descriptive name
- **Purpose**: Just for identification/organization
- **Example**: `LoRA-Mail-Assistant-M3-Mac`
- **Note**: You can have multiple tokens with different names

#### Token Type (READ vs WRITE)

**For LoRA-Mail Assistant, you need:**

✅ **READ Token** (Recommended)
- **What it does**: Allows downloading models from Hugging Face
- **Why this is enough**: You only need to download Llama 3.2 3B, not upload
- **Security**: More secure, can't accidentally modify anything
- **Select**: Choose **"Read"** token type

⚠️ **WRITE Token** (Only if needed)
- **What it does**: Allows uploading models, creating repos, etc.
- **When needed**: Only if you plan to upload your fine-tuned model to Hugging Face
- **Security**: Less secure, has more permissions
- **For this project**: **NOT NEEDED** - stick with Read token

#### Token Permissions (if using WRITE token)

If you selected WRITE token, you'll see additional options:

- **Repository permissions**: 
  - `repo` - Full repository access
  - `repo:read` - Read repository access
  - `repo:write` - Write repository access
  
**For LoRA-Mail Assistant**: You don't need WRITE token, so skip this.

### Step 4: Generate Token

1. Click **"Generate token"** button
2. **IMPORTANT**: Copy the token immediately!
   - The token will be shown **ONCE**
   - It looks like: `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
   - Starts with `hf_` followed by long string
   - **You cannot view it again** after closing the page

### Step 5: Save Token Securely

**Option 1: Copy to clipboard**
- Copy the token
- Store it securely (password manager, notes app, etc.)

**Option 2: Save to file** (for local use only)
```bash
# Create a .env file in your project
echo "HF_TOKEN=your_token_here" > .env
```

**Option 3: Use huggingface-cli login** (Recommended)
- Don't copy the token manually
- Use the command line tool instead (see below)

## Using the Token

### Method 1: Command Line Login (Recommended)

```bash
huggingface-cli login
```

When prompted:
1. Paste your token
2. Press Enter
3. Token is saved to `~/.huggingface/token`

**Advantages**:
- ✅ Secure (stored in your home directory)
- ✅ Works automatically with transformers library
- ✅ No need to manually set environment variables

### Method 2: Environment Variable

```bash
export HF_TOKEN=your_token_here
```

Or add to your shell profile (`~/.zshrc` or `~/.bashrc`):
```bash
echo 'export HF_TOKEN=your_token_here' >> ~/.zshrc
source ~/.zshrc
```

### Method 3: Python Code (Not Recommended)

```python
from huggingface_hub import login
login(token="your_token_here")
```

## Token Details Summary

### For LoRA-Mail Assistant:

| Setting | Value | Notes |
|---------|-------|-------|
| **Token Name** | `lora-mail-assistant` | Any descriptive name |
| **Token Type** | **Read** | ✅ Recommended |
| **Permissions** | Read-only | Can download models |
| **Expiration** | No expiration (default) | Or set custom date |
| **Token Format** | `hf_xxxxxxxxxxxxx` | Starts with `hf_` |

### What the Token Allows

✅ **Can do:**
- Download models (Llama 3.2 3B)
- Access gated models (Llama requires authentication)
- Use with transformers library
- Use with huggingface_hub library

❌ **Cannot do** (with Read token):
- Upload models
- Create repositories
- Modify existing repos
- Delete content

## Security Best Practices

1. **Never commit tokens to Git**
   - Add `.env` to `.gitignore` ✅ (already done)
   - Never paste tokens in code
   - Never share tokens publicly

2. **Use Read tokens when possible**
   - More secure
   - Sufficient for downloading models

3. **Revoke unused tokens**
   - Go to settings/tokens
   - Click "Revoke" on unused tokens

4. **Set expiration** (optional)
   - For extra security
   - Can create new token when needed

## Troubleshooting

### "Token not found" or "Invalid token"
- Check you copied the full token (starts with `hf_`)
- Verify token hasn't been revoked
- Make sure you're using the correct token

### "Authentication required"
- Run: `huggingface-cli login`
- Or set: `export HF_TOKEN=your_token`

### "401 Unauthorized"
- Token might be expired or revoked
- Create a new token
- Re-authenticate

### "Model not found" (even with token)
- Some models require accepting terms first
- Visit model page: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
- Click "Agree and access repository"
- Then use your token

## Quick Reference

**Token Creation Checklist:**
- [ ] Go to https://huggingface.co/settings/tokens
- [ ] Click "New token"
- [ ] Name: `lora-mail-assistant`
- [ ] Type: **Read**
- [ ] Click "Generate token"
- [ ] Copy token immediately
- [ ] Run: `huggingface-cli login`
- [ ] Paste token when prompted
- [ ] Verify: `huggingface-cli whoami`

**Token Format:**
```
hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**After Setup:**
```bash
# Verify authentication
huggingface-cli whoami

# Should show your username
```

## Next Steps

After creating and saving your token:

1. **Verify setup:**
   ```bash
   python scripts/check_setup.py
   ```

2. **Start training:**
   ```bash
   python -m src.train
   ```

The token will be used automatically when downloading Llama 3.2 3B!

