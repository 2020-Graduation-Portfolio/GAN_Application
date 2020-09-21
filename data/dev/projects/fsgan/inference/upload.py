import os
from flask import Flask, current_app, request, redirect, url_for, render_template
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_ngrok import run_with_ngrok
from fsgan.inference import face_swap_images2images

app = Flask(__name__)
run_with_ngrok(app)   # starts ngrok when the app is run

UPLOADS_DEFAULT_DEST = 'static/images'
SOURCE_DEST = os.path.join(UPLOADS_DEFAULT_DEST, 'source')
TARGET_DEST = os.path.join(UPLOADS_DEFAULT_DEST, 'target')
RESULT_DEST = os.path.join(UPLOADS_DEFAULT_DEST, 'result')
app.config['UPLOADS_DEFAULT_DEST'] = UPLOADS_DEFAULT_DEST
app.config['RESULT_DEST'] = RESULT_DEST

sources = UploadSet('source', IMAGES)
targets = UploadSet('target', IMAGES)
configure_uploads(app, (sources, targets))


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        source_name = sources.save(request.files.get('source'))
        target_name = targets.save(request.files.get('target'))
        result_name = '_'.join([os.path.splitext(source_name)[0], os.path.splitext(target_name)[0]]) + '.jpg'
        result_image_path = os.path.join(RESULT_DEST, result_name)

        face_swap_images2images.main(source_path=SOURCE_DEST, target_path=TARGET_DEST, output_path=RESULT_DEST)
        return render_template('result.html', imagepath=result_image_path)
    return render_template('upload.html')


if __name__ == '__main__':
    app.run()
