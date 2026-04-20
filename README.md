# NTHU CCXP Captcha API

Minimal Vercel-ready Python API for captcha inference.

## Endpoint

- `POST /api/decaptcha`

Request body:

```json
{
  "img": [137, 80, 78, 71]
}
```

Response body:

```json
{
  "answer": "123456"
}
```

## Deploy To Vercel

1. Install Vercel CLI.
2. Run `vercel login`.
3. Run `vercel`.
4. Deploy with `vercel --prod`.

## Runtime Files

- `api/main.py`
- `api/decaptcha.pt`
- `requirements.txt`
