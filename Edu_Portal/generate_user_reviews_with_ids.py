"""
This script generates random user reviews for a set of schools and saves the data to a CSV file.
Each school is assigned a random review and rating, simulating user feedback.

Steps:
1. A predefined set of school IDs and sample reviews are used.
2. For each school ID, a random review is selected and a random rating between 1 and 5 is generated.
3. The data (user ID, school ID, review, and rating) is stored in a pandas DataFrame.
4. The DataFrame is saved to a CSV file called 'generate_user_reviews_with_ids.csv'.

The generated data can be used for training recommendation models or other data analysis tasks.

Dependencies:
- pandas for data manipulation and file saving.
- random for generating random reviews and ratings.
"""


import pandas as pd
import random
import os

# Provided school ID and sample reviews
data = {
    "school_id": [
        700910011, 700400393, 700121210, 700350561, 700915064, 700400277, 700320291,
        700231522, 700231530, 700211276, 700320937, 700152033, 700400391, 700910158,
        700910169, 700221474, 700350595, 700211300, 700320366, 700270645, 700220574,
        700111740, 700152058, 700400031, 700400179, 700400178, 700400180, 700400212,
        700251306, 700400010, 700400423, 700400424, 700400149, 700910274, 700910276,
        700321927, 700914251, 700251363, 700320457, 700160028, 700220608, 700330837,
        700340570, 700260745
    ],

    # Sample reviews for generating relevant user feedback
    "sample_reviews": [
        "The teachers are very supportive and dedicated.",
        "The school has great facilities, but the administration could improve.",
        "I love the extracurricular activities offered here.",
        "The academic performance of the school is top-notch!",
        "My child enjoys going to this school every day.",
        "There are some issues with overcrowding in classrooms.",
        "The school provides a safe and nurturing environment.",
        "I appreciate the diversity and inclusivity at this school.",
        "There are plenty of resources available for students.",
        "The school's communication with parents is very effective.",
        "The sports program is excellent and well-managed.",
        "Staff are friendly and approachable.",
        "The curriculum is challenging and engaging.",
        "I wish there were more advanced placement options.",
        "The school community is very active and involved.",
        "Some teachers need to improve their teaching methods.",
        "The school has a good reputation in the area.",
        "I would recommend this school to other parents.",
        "There are frequent updates about school events.",
        "The cafeteria food could be better.",
        "Overall, a great place for learning and growth."
    ]
}

# Generate user reviews based on school IDs
generate_user_reviews_with_ids = []
for school_id in data["school_id"]:
    review = random.choice(data["sample_reviews"])
    rating = round(random.uniform(1, 5), 1)  # Rating between 1 and 5
    generate_user_reviews_with_ids.append({
        'user_id': random.randint(1, 20),
        'school_id': school_id,
        'review': review,
        'rating': rating
    })

# Create DataFrame from user_reviews
df = pd.DataFrame(generate_user_reviews_with_ids)

# Save DataFrame to CSV
df.to_csv('generate_user_reviews_with_ids.csv', index=False)

print("Data saved to 'generate_user_reviews_with_ids.csv'")
