from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

filename = 'model.joblib'
model = joblib.load(filename)

@app.route('/predict', methods=['POST'])
def predict_study_time():
    # Predicts study time for a list of student data dictionaries with topics.
    # Args:
    #     student_data_list: A list of dictionaries, where each dictionary 
    #                        represents a student and contains 'accuracy', 
    #                        'quiz_time_taken', and 'topic' keys.

    # Returns:
    #     A list of dictionaries, where each dict contains the topic and the 
    #     predicted study time in minutes.
    # 

    try:
        data = request.get_json()
        results = []
        for student_data in data:
            if not isinstance(student_data, dict) or 'quiz_score' not in student_data or 'quiz_time_taken' not in student_data or 'topic' not in student_data:
                return jsonify({'error': 'invalid data'}), 400
            try:
                new_student_data = pd.DataFrame({
                    'quiz_score': [student_data['quiz_score']],
                    'quiz_time_taken': [student_data['quiz_time_taken']]
                })
                hours = model.predict(new_student_data)
                hours = hours.round(1)
                results.append({
                    'topic': student_data['topic'], 
                    'study_duration': hours[0]
                })
            except (ValueError, TypeError) as e:
                return jsonify({'error': f'Error processing data: {str(e)}'}), 400

        return jsonify(results), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
           
if __name__ == '__main__':
    app.run(debug=True)
