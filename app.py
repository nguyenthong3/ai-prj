from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import time
from os.path import join, dirname, realpath
from werkzeug.utils import secure_filename

from keras.preprocessing import image
from keras import models
import numpy as np
import joblib

UPLOAD_FOLDER = join(dirname(realpath(__file__)), 'static\\uploads\\')
MODEL_AGE_FILE_PATH = join(dirname(realpath(__file__)), 'knn_model_age.pkl')
MODEL_GENDER_FILE_PATH = join(dirname(realpath(__file__)), 'knn_model_gender.pkl')

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

gen_list = {
    0: 'Nam',
    1: 'Nữ',
}
gend = ['Nam','Nữ']
app = Flask(__name__)

app.secret_key = "thong1412"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16*1024*1024


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:

  def __init__(self,X_train, y_train,k):
    self.X_train = X_train
    self.y_train = y_train
    self.k = k
    
  # Hàm tìm k điểm gần nhất
  def __get_k_neighbors(self, X_test, k):
      distances = [euclidean_distance(X_test, x) for x in self.X_train]
      sorted_distances = np.argsort(distances)
      return sorted_distances[:self.k]

  # Hàm dự đoán
  def predict(self, X_test):
        y_predict = []
        distances_arr = []
        k_nearest_neighbors = self.__get_k_neighbors(X_test,self.k)

        for i in k_nearest_neighbors:
            y_predict.append(self.y_train[i])
        
        return np.sum(y_predict)/self.k
  
  def score():
    print 


@app.route('/', methods=['GET'])
def to_homepage():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def handle_image():
    if 'file' not in request.files:
        flash('No file part!')
        return redirect(request.url)
    file = request.files['file']

    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filePath = join(app.config['UPLOAD_FOLDER'], filename).replace('\\','/')
        file.save(filePath)
        flash('Image successfully uploaded and calculate ...')

        img = image.image_utils.load_img(filePath, target_size=(48,48))
        # imgArr = image.image_utils.img_to_array(img)
        x = np.array(img)
        #convert to grayscale       
        X = []
        R, G, B = x[:,:,0], x[:,:,1], x[:,:,2]
        X = 0.3 * R + 0.6 * G + 0.1 * B
        X = np.array(X)
        x = image.image_utils.img_to_array(X)           #convert image to np array
        x = np.expand_dims(x, axis=0)         #Thêm một axis mới

        images = np.vstack([x])

        knnAge = joblib.load(MODEL_AGE_FILE_PATH)
        predAge = knnAge.predict(images.reshape(-1))
        age = np.around(predAge)

        knnGender = joblib.load(MODEL_GENDER_FILE_PATH)
        predGender = knnGender.predict(images.reshape(-1))
        gender = gen_list[np.around(predGender)]
        
        # gender = age_list[np.around(pred[0])[0,0]]
        return render_template('index.html', filename=filename, age=age, gender=gender)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(port=5001, debug=True)
