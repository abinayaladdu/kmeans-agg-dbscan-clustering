from flask import Flask, render_template, request, make_response, jsonify
import pickle
import io
import csv
from io import StringIO
import pandas as pd
from sklearn.metrics import silhouette_score
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


global s,s1,s2
app = Flask(__name__)
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route('/defaults',methods=['POST'])


def defaults():


    return render_template('index.html')



def transform(text_file_contents):


    return text_file_contents.replace("=", ",")




@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        SepalLengthCm = float(request.form['SepalLengthCm'])
        SepalWidthCm = float(request.form['SepalWidthCm'])
        PetalLengthCm = float(request.form['PetalLengthCm'])
        PetalWidthCm = float(request.form['PetalWidthCm'])


        if request.form['cls'] == 'kmeans':
            df = pd.read_csv("Irisdata.csv")
            loaded_model = pickle.load(open('kmeans.pkl', 'rb'))
            kmeans = loaded_model.fit(df)
            prediction = kmeans.predict([[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]])
           
        elif request.form['cls'] == 'agglomerative':
          df = pd.read_csv("Irisdata.csv")        
          loaded_model = pickle.load(open('agglomerative.pkl', 'rb'))
          prediction = loaded_model.fit_predict([[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]])


        elif request.form['cls'] == 'dbscan':
            df = pd.read_csv("Irisdata.csv")
            loaded_model = pickle.load(open('dbscan.pkl', 'rb'))
            prediction = loaded_model.fit_predict([[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]])


    return render_template('index.html',prediction_text =" {} cluster".format(prediction))


@app.route('/default',methods=["POST"])


def default():


    return render_template('layout.html')




@app.route('/transform', methods=["POST"])


def transform_view():


    f = request.files['data_file']


    if not f:


        return "No file"




    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)


    csv_input = csv.reader(stream)


    print(csv_input)


    for row in csv_input:


        print(row)




    stream.seek(0)


    result = transform(stream.read())


   
    if request.form['cls'] == 'kmeans':
        df = pd.read_csv(StringIO(result))
   
        loaded_model = pickle.load(open('kmeans.pkl', 'rb'))
        kmeans = loaded_model.fit(df)
        df['Cluster'] = kmeans.predict(df)
        response = make_response(df.to_csv())
        response.headers["Content-Disposition"] = "attachment; filename=KMEANS.csv"
       


       
    elif request.form['cls'] == 'agglomerative':
        df = pd.read_csv(StringIO(result))
       
        loaded_model = pickle.load(open('agglomerative.pkl', 'rb'))


        df['Cluster'] = loaded_model.fit_predict(df)
        response = make_response(df.to_csv())
        response.headers["Content-Disposition"] = "attachment; filename=AGGLOMERATIVE.csv"


    elif request.form['cls'] == 'dbscan':
        df = pd.read_csv(StringIO(result))
   
        model = pickle.load(open('dbscan.pkl', 'rb'))
        labels = model.labels_
        df['Cluster'] = model.fit_predict(df)
        response = make_response(df.to_csv())


        response.headers["Content-Disposition"] = "attachment; filename=DBSCAN.csv"
       
    return response


@app.route('/transform1', methods=["POST"])


def transform_view1():


    f = request.files['data_file']


    if not f:


        return "No file"




    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)


    csv_input = csv.reader(stream)


    print(csv_input)


    for row in csv_input:


        print(row)




    stream.seek(0)


    result = transform(stream.read())


   
    if request.form['clas'] == 'kmeans1':
        df = pd.read_csv(StringIO(result))
   
        loaded_model = pickle.load(open('kmeans.pkl', 'rb'))
        kmeans = loaded_model.fit(df)
        cluster = kmeans.predict(df)
        s = silhouette_score(df, kmeans.labels_)
        center = Counter(kmeans.labels_)
        fig = plt.figure(figsize=(5, 4))
        plt.scatter(df['PetalLengthCm'], df['PetalWidthCm'],c=cluster, cmap='Paired')
        plt.title('KMeans')
        imgdata = StringIO()
        fig.savefig(imgdata, format='svg')
        imgdata.seek(0)
        data = imgdata.getvalue()
       
    elif request.form['clas'] == 'agglomerative2':
        df = pd.read_csv(StringIO(result))
       
        loaded_model = pickle.load(open('agglomerative.pkl', 'rb'))


        cluster = loaded_model.fit_predict(df)
        s = silhouette_score(df, loaded_model.labels_)
        center = Counter(loaded_model.labels_)
        fig = plt.figure(figsize=(5, 4))
        plt.scatter(df['PetalLengthCm'], df['PetalWidthCm'],c=cluster, cmap='Paired')
        plt.title('Agglomerative')
        imgdata = StringIO()
        fig.savefig(imgdata, format='svg')
        imgdata.seek(0)
        data = imgdata.getvalue()
   
    elif request.form['clas'] == 'dbscan3':
        df = pd.read_csv(StringIO(result))
   
        loaded_model = pickle.load(open('dbscan.pkl', 'rb'))
        labels = loaded_model.labels_
        cluster = loaded_model.fit_predict(df)
        s = silhouette_score(df, labels)
        center = Counter(loaded_model.labels_)
        fig = plt.figure(figsize=(5, 4))
        plt.scatter(df['PetalLengthCm'], df['PetalWidthCm'],c=cluster, cmap='Paired')
        plt.title('DBSCAN')
        imgdata = StringIO()
        fig.savefig(imgdata, format='svg')
        imgdata.seek(0)
        data = imgdata.getvalue()
    return render_template('layout.html', prediction =" The Silhoutee score is  {}".format(s), prediction1= "The counters of the clusters are {}".format(center), image=data)




if __name__=="__main__":
    app.run(debug=True)

