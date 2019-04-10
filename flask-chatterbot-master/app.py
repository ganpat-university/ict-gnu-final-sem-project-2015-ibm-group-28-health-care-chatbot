
from flask import Flask,render_template, request, Response, session, redirect, url_for
from weka.classifiers import PredictionOutput
from weka.classifiers import Evaluation
from weka.core.classes import Random
import weka.core.jvm as jvm
import weka.core.serialization as serialization
from weka.classifiers import Classifier
from weka.core.converters import Loader
import pandas as pd
from io import StringIO
import json

app = Flask(__name__)

@app.route('/bot',methods=['GET','POST'])
def index():
    if request.method == "GET":
        return render_template('bot.html')
    if request.method == "POST":
        # jvm.stop()
        jvm.start()
        f = open("instances.arff", "a")
        args = request.form.to_dict()
        weight_lb = float(args['weight'])*2.20462
        bmi = (weight_lb/pow(float(args['height']),2))*703
        hypertensive_status = args['hypertensive_status']
        heart_disease_status = args['heart_disease_status']
        if heart_disease_status =="Yes":
            heart_disease_status = '1'
        else:
            heart_disease_status='0'
        if hypertensive_status == "Yes":
            hypertensive_status = '1'
        else:
            hypertensive_status='0'
        
        st = "\n"+args['gender']+","+args['age']+","+hypertensive_status+","+heart_disease_status+","+args['marrital_status'] + \
            ","+args['work_type']+","+args['residence']+"," + \
            args['hypertension']+","+str(bmi)+",'"+args['smoking_status'].lower()+"',?"
        print(st)
        f.write(st)
        f.close()
        objects = serialization.read_all("J48.model")
        loader = Loader(classname="weka.core.converters.ArffLoader")
        csr = Classifier(jobject=objects[0])
        output_results = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV")
        data1 = loader.load_file("instances.arff")
        data1.class_is_last()
        ev2 = Evaluation(data1)
        ev2.test_model(csr, data1, output_results)
        
        TESTDATA = StringIO("Instance,Actual,Predicted," +
                            output_results.buffer_content())
        df = pd.read_csv(TESTDATA)
        prediction = list(df.Predicted).pop().split(":")[1]
        print(prediction)
        # jvm.stop()
        response  = {
            "status":"200",
            "prediction":prediction
        }
        return Response(json.dumps(response,indent=2),mimetype="application/json")

if __name__ == '__main__':

    app.run(debug=True)
