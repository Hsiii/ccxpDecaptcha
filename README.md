# NTHU CCXP Decaptcha

Automatically fill in the captcha on the NTHU Academic Information Systems.

## Download & Installation

- [Firefox Add-on](https://addons.mozilla.org/zh-TW/firefox/addon/nthu-ccxp-decaptcha/)
- [Chrome Extension](https://chrome.google.com/webstore/detail/nthu-ccxp-decaptcha/hpbhebpkmhpeoomcmdmlmhlclhbbdjho?hl=zh-TW)

## Privacy Policy

This extension does not collect any personal data.

## Implementation Details

### Data Collection

- Download and manually label the images
- Save repeated renders of the same `pwdstr` together so they can be grouped into the same split

### Data Preprocessing

- Keep the full captcha image at its native size
- Build grouped train/validation splits by captcha identity instead of by digit crop
- Apply live-site-matched augmentation during training

### Model

- Lightweight CNN backbone over the full captcha image
- Adaptive pooling to 6 positions
- Six classification heads, one per digit
- Exact 6-digit match rate as the primary validation metric
- Export both the best float checkpoint and an `int8` inference artifact

### Extension

- Download the captcha image
- Transform to an Array of bytes
- JSON.stringify the array and pass it to the API
