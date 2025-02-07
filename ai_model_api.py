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
            if not isinstance(student_data, dict) or 'accuracy' not in student_data or 'quiz_time_taken' not in student_data or 'topic' not in student_data:
                return jsonify({'error': 'invalid data'}), 400
            try:
                new_student_data = pd.DataFrame({
                    'accuracy': [student_data['accuracy']],
                    'quiz_time_taken': [student_data['quiz_time_taken']]
                })
                minutes = model.predict(new_student_data)
                minutes = minutes.round(0)
                results.append({
                    'topic': student_data['topic'], 
                    'study_duration': int(minutes[0])
                })
            except (ValueError, TypeError) as e:
                return jsonify({'error': f'Error processing data: {str(e)}'}), 400

        return jsonify(results), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
           
if __name__ == '__main__':
    app.run(debug=True)

# student_data_list = [
#     {'accuracy': 0.3, 'quiz_time_taken': 20, 'topic': 'Energy'},
#     {'accuracy': 0.7, 'quiz_time_taken': 15, 'topic': 'Simple Machines'},
#     {'accuracy': 0.5, 'quiz_time_taken': 25, 'topic': 'Motion and Forces'},
# ]

# predictions = predict_study_time(student_data_list)

# print(predictions)