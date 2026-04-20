# ccxpDecaptcha

Minimal Vercel Python API for NTHU academic information system ([CCXP](https://www.ccxp.nthu.edu.tw/ccxp/INQUIRE/)) decaptcha.  
The API is optimized for Vercel cold starts by using a low-dependency custom inference runtime (without PyTorch at runtime), with model weights trained by [25349023](https://github.com/25349023).  
This project is forked from [25349023/nthu-ccxp-captcha](https://github.com/25349023/nthu-ccxp-captcha/) and remains licensed under MIT.

## Endpoint

- `POST /api/decaptcha`

Request body:

```http
Content-Type: application/octet-stream

<raw captcha image bytes>
```

Response body:

```json
{
  "answer": "123456"
}
```
