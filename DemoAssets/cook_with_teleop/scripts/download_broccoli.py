"""Download low-poly broccoli OBJ from Poly Pizza."""
import re
import urllib.request
import os
import sys

TARGET = os.path.join(os.path.dirname(__file__), "..", "broccoli.obj")

page_url = "https://poly.pizza/m/e2Z3XDxtT41"
try:
    req = urllib.request.Request(page_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
except Exception as e:
    print(f"Failed to fetch page: {e}")
    sys.exit(1)

# Look for OBJ/GLTF/GLB download URLs in the HTML/JS
patterns = [
    r'(https?://[^\s"\'<>]+\.obj)',
    r'(https?://[^\s"\'<>]+\.gltf)',
    r'(https?://[^\s"\'<>]+\.glb)',
    r'(https?://[^\s"\'<>]+poly\.googleapis\.com[^\s"\'<>]*)',
    r'(https?://[^\s"\'<>]*storage\.googleapis\.com[^\s"\'<>]*)',
]
found = []
for pat in patterns:
    found.extend(re.findall(pat, html, re.IGNORECASE))

if found:
    print("Found model URLs:")
    for u in found[:10]:
        print(f"  {u}")
    obj_urls = [u for u in found if u.lower().endswith(".obj")]
    if obj_urls:
        url = obj_urls[0]
        print(f"\nDownloading OBJ: {url}")
        urllib.request.urlretrieve(url, TARGET)
        print(f"Saved to {TARGET}")
    else:
        print("\nNo direct OBJ URL found. Trying first URL...")
        url = found[0]
        urllib.request.urlretrieve(url, TARGET)
        print(f"Saved to {TARGET}")
else:
    print("No model URLs found in page HTML.")
    print("The page likely loads models via JavaScript.")
    print(f"\nPlease manually download from: {page_url}")
    print(f"Save the OBJ file to: {TARGET}")
