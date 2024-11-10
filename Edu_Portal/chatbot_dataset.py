"""
This Python script is designed to expand a chatbot's training dataset by generating a larger set of question-answer pairs. It starts by 
defining a set of additional questions and corresponding answers related to school recommendations. Then, it multiplies these pairs to 
simulate a larger dataset, further modifying the questions by introducing variations using a predefined list of terms. This process ensures 
greater diversity in the dataset, which can improve the chatbot's ability to understand and respond to a wider range of queries. Finally, 
the expanded dataset is saved to a CSV file for use in training the chatbot.

"""




#chatbot_dataset.py
import pandas as pd

# Additional question variations to further diversify the dataset
additional_questions = [
    "How do I find schools with good extracurricular activities?",
    "Can the recommendation system help me find a school with a strong sports program?",
    "What are the factors that determine the quality of a school?",
    "Can I get recommendations for schools based on academic performance alone?",
    "How do I know if a school is a good fit for my child?",
    "What is the best way to compare different schools?",
    "How does the recommendation system assess school facilities?",
    "Are schools with a high student-teacher ratio recommended?",
    "How does the recommendation system work for students with special needs?",
    "What is the role of school diversity in the recommendation system?",
    "How can I filter schools based on location?",
    "How can I choose a school based on its reputation?",
    "Can I get a recommendation for a school with good reviews?",
    "Do school facilities like sports fields matter in school recommendations?",
    "How important is a school's academic ranking in the recommendation system?",
    "Can I find schools with a good reputation for technology?",
    "How do I get recommendations for private schools?",
    "What is the importance of school location in the recommendation system?",
    "Can the recommendation system suggest schools based on school type?",
    "How do I find a school with strong arts and culture programs?",
    "Are online reviews important for school recommendations?",
    "Can I find a school based on class size or teacher-to-student ratio?",
    "How do I choose a school based on student satisfaction ratings?",
    "Can the system help me find a school that promotes gender equality?",
    "How do I know if a school has a safe and inclusive environment?",
    "What schools are recommended for children interested in STEM fields?",
    "Can the recommendation system suggest a school based on affordability?",
    "How does the recommendation system handle schools with online courses?",
    "Can I get a recommendation for a school based on school safety?",
    "How do I find a school with excellent facilities?",
    "How do I search for schools with a modern curriculum?",
    "What are the best schools for children with learning disabilities?",
    "Can I find a school that offers bilingual education?",
    "Do school ratings affect the recommendations?",
    "Can I get recommendations for international schools?",
    "Are there specific questions I should ask when choosing a school?"
]

additional_answers = [
    "The recommendation system can help you find schools with strong extracurricular programs based on user reviews and data analysis.",
    "Yes, the system can help you find schools with excellent sports programs by analyzing available data on school activities.",
    "Factors like academic performance, school reviews, student-teacher ratio, facilities, and location are considered.",
    "Yes, if academic performance is a top priority, the system can recommend schools based on their performance ratings.",
    "To know if a school is a good fit, consider academic performance, location, reviews, and your child's needs.",
    "The best way to compare schools is to consider factors like academic performance, reviews, location, and facilities.",
    "The system assesses school facilities such as libraries, science labs, and sports facilities to make recommendations.",
    "Schools with a lower student-teacher ratio tend to be recommended for their ability to provide personalized attention.",
    "The system can recommend schools based on the needs of special education students, taking into account available resources.",
    "Diversity in schools is an important factor in recommendations, especially for families seeking an inclusive environment.",
    "You can filter schools by location, proximity to your home, and preferred areas using the recommendation system.",
    "School reputation plays a key role in recommendations, as well-established schools tend to be more highly rated.",
    "The system can recommend schools based on their reviews, which give valuable insights into the quality of the school.",
    "Yes, sports fields and extracurricular facilities like sports fields are considered in school recommendations.",
    "Academic ranking is one of the important factors, but the recommendation system also considers reviews, location, and facilities.",
    "Schools that are known for their reputation in technology or innovation can be recommended by the system.",
    "If you prefer private schools, the system can recommend them based on factors like fees, reputation, and reviews.",
    "Location is very important in the recommendation process, especially if proximity to home or work is a priority.",
    "Yes, the system can recommend schools based on specific school types such as public, private, or charter schools.",
    "You can get recommendations for schools with arts programs by specifying this preference in the system.",
    "Yes, online reviews and ratings are an essential part of the recommendation system, giving insight into school quality.",
    "Yes, class size and teacher-to-student ratio are key factors in determining the right fit for your child.",
    "Student satisfaction ratings are considered to help recommend schools where students are generally content with their experience.",
    "Yes, gender equality and inclusivity are key factors in the recommendation system, especially for schools that promote diversity.",
    "To ensure safety and inclusivity, the system can recommend schools based on available reports and data on these aspects.",
    "The system can suggest schools based on their reputation in STEM fields, provided that this is one of your preferences.",
    "Yes, the recommendation system takes into account school affordability, which includes both fees and available financial aid.",
    "Yes, the system can recommend online schools or schools offering online courses if that's one of your preferences.",
    "School safety is considered based on available safety records, reviews, and any relevant data on the school's security measures.",
    "You can find schools with excellent facilities by using filters in the recommendation system to specify your preferences.",
    "The system can help you search for schools with a modern curriculum focused on current trends in education.",
    "The recommendation system can suggest schools that specialize in helping children with learning disabilities or provide specialized support.",
    "Yes, the system can recommend bilingual schools or those with language immersion programs.",
    "Yes, school ratings from various review platforms are taken into account when recommending schools.",
    "International schools can be recommended based on location, type, and your specific requirements for international curricula.",
    "Some key questions to ask include academic performance, student-teacher ratio, school facilities, and the school's approach to extracurriculars."
]

# To simulate an even larger dataset, we'll repeat and add more variations
expanded_questions = additional_questions * 60  # Repeat the list 60 times
expanded_answers = additional_answers * 60  # Repeat the list 60 times

# Adding even more random variations to questions
variations = ['recommendations', 'find', 'top', 'best', 'school', 'academics', 'reviews', 'extracurricular', 'location', 'performance', 'safety', 'modern', 'curriculum', 'extracurricular activities', 'teacher-student ratio', 'affordable', 'private', 'public', 'gender equality', 'learning disabilities', 'bilingual', 'sports', 'arts', 'technology', 'student satisfaction', 'student-teacher ratio']

expanded_questions_with_variations = []
for question in expanded_questions:
    modified_question = question
    for word in variations:
        modified_question = modified_question.replace('school', word)  # Introduce slight variations
    expanded_questions_with_variations.append(modified_question)

# Create the final DataFrame for the even larger dataset
df = pd.DataFrame({
    'question': expanded_questions_with_variations,
    'answer': expanded_answers
})

# Save the further expanded dataset to a CSV file
df.to_csv('chatbot_training_data_expanded_v3.csv', index=False)
print("Further expanded training dataset with additional variations created!")
