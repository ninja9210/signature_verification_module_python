# Signature Verification Module

A Python module that checks whether a handwritten signature is GENUINE or FORGED
by comparing it against 3-5 reference signatures of the same user.

It works on image file paths only. It has no web, database, or Django dependency,
so it can be called from any backend.

## 1. Requirements

- Python 3.9 or newer
- OpenCV 4.x (OpenCV 5.x is NOT supported: it does not include `HOGDescriptor`)
- NumPy

## 2. Installation

1. Copy the `signature_verification/` folder into your project, next to `manage.py`:

       your_project/
       ├── project folders and files...
       ├── signature_verification/
       └── ...

2. Install the dependencies:

       pip install -r signature_verification/requirements.txt

3. Check that the install works:

       python -c "import cv2; print(cv2.__version__, hasattr(cv2, 'HOGDescriptor'))"

   The version should start with 4 and the second value should be `True`.
   If it prints version 5.x or `False`, run:

       pip install "opencv-python<5"

## 3. Usage

    from signature_verification import verify_signature

    result = verify_signature(
        ["media/user1/ref1.png", "media/user1/ref2.png", "media/user1/ref3.png"],
        "media/uploads/new_signature.png",
    )
    print(result)

### Parameters

| Parameter  | Type        | Description |
|------------|-------------|-------------|
| `refs`     | list of str | File system paths of 3 to 5 genuine signature images of the same user |
| `new_path` | str         | File system path of the signature image to check |

Paths must be real file paths (for example `uploaded_file.path` in Django), not URLs.

### Return value

The function always returns a dictionary and never raises an exception.

Success:

    {"similarity": 84, "result": "GENUINE", "references_used": 5, "error": None}

Failure (missing file, blank image, unreadable image):

    {"similarity": None, "result": None, "references_used": 0, "error": "message"}

| Key               | Meaning |
|-------------------|---------|
| `similarity`      | Score from 0 to 100. It is an indicator, not a true probability. About 50 is the decision boundary. |
| `result`          | `"GENUINE"` or `"FORGED"`. Use this for the decision. |
| `references_used` | Number of reference images used |
| `error`           | `None` on success, otherwise an error message to show the user |

## 4a. Django example

    # views.py
    from rest_framework.decorators import api_view
    from rest_framework.response import Response
    from signature_verification import verify_signature

    @api_view(["POST"])
    def verify(request):
        user = request.user
        refs = [s.image.path for s in user.reference_signatures.all()]   # 3-5 saved at registration
        new_file = SignatureUpload.objects.create(user=user, image=request.FILES["image"])

        result = verify_signature(refs, new_file.image.path)

        if result["error"]:
            return Response({"error": result["error"]}, status=400)
        return Response(result)

## 4b. Flask example

    import os, tempfile
    from flask import Flask, request, jsonify
    from signature_verification import verify_signature

    app = Flask(__name__)

    @app.route("/verify", methods=["POST"])
    def verify():
        refs = [f for f in request.files.getlist("refs") if f.filename]
        new = request.files.get("new")
        if len(refs) < 3 or not new:
            return jsonify({"error": "Upload 3-5 references and 1 new signature."}), 400

        with tempfile.TemporaryDirectory() as tmp:
            ref_paths = []
            for i, f in enumerate(refs[:5]):
                p = os.path.join(tmp, f"ref{i}.png")
                f.save(p)
                ref_paths.append(p)
            new_path = os.path.join(tmp, "new.png")
            new.save(new_path)
            result = verify_signature(ref_paths, new_path)

        return jsonify(result), (400 if result["error"] else 200)

The module itself is framework independent. It only needs file system paths,
so it works the same way with Flask, Django, or FastAPI.

## 5. What the backend must do

1. At registration, save 3 to 5 genuine signatures per user.
2. At verification, pass those saved paths and the uploaded image path to `verify_signature`.
3. If `error` is not `None`, show the message to the user.
4. Store `result` and `similarity` in the verification history.

## 6. Testing from the command line

From the folder that contains `signature_verification/`:

    python -c "from signature_verification import verify_signature; print(verify_signature(['a1.png','a2.png','a3.png'], 'new.png'))"

## 7. How it works

1. Preprocessing: grayscale, blur, Otsu threshold, remove printed text and specks, crop, resize to 256x128
2. Alignment by centroid
3. Five comparison measures: overlap (IoU), projection profiles, HOG, grid density, chamfer distance
4. Scores are normalized by how much this user's own reference signatures vary
5. The combined score is compared against a fixed threshold

## 8. Accuracy and limitations

- About 75% accuracy on a subset of the CEDAR dataset with 5 references.
- Roughly 25% of forged signatures may pass and 28% of genuine ones may be rejected.
- Weak against skilled forgeries.
- Needs 3-5 reference signatures. Single-reference mode exists but is untuned and unreliable.
- Best results come from clear scans or photos: dark pen, white paper, no shadows.
- The result should support a human decision, not replace it, for any high-stakes use.

## 9. Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: signature_verification` | The folder must be in your project root, or add it to `PYTHONPATH` |
| `AttributeError: cv2 has no attribute HOGDescriptor` | Run `pip install "opencv-python<5"` |
| `error: Image blank` | The image has no visible signature. Ask the user to upload a clearer image |
| `error: ... could not be read` | Wrong path or unsupported file. Check that the file exists and is PNG or JPG |
| Many genuine signatures rejected | Ask for 5 references instead of 3, with clear images |