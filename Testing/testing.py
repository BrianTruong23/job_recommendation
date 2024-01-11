
import pandas as pd 


postings = ["job_posting.csv", "data_science_roles_testing.csv", "web_dev_roles_testing.csv" ]


def print_out_results(posting):
 

    roles = pd.read_csv(posting)
    # Provided data
    total_roles = len(roles)
    data_science_roles = 0
    web_roles = 0
    # Loop through the roles and count those related to data science and web development
    for role in roles["Role"]:
        if "Data Science" in role or "Data" in role or "Data Scientist" in role:
            data_science_roles += 1

    for role in roles["Role"]:
        if "Angular" in role or "Java" in role or  "Javascript" in role or "Backend" in role or "Software Development" in role or "Full Stack" in role or "Frontend" in role:
            web_roles += 1

    print("For posting csv: ", posting)

    if posting == "job_posting.csv":
        # Print the results
        print("Total Roles:", total_roles)
        print("Data Science Roles:", data_science_roles)

        print("Total Roles:", total_roles)
        print("Web Dev Roles:", web_roles)
    elif "data_science" in posting:
        percentage_correct = data_science_roles / total_roles * 100 
        print("Total Roles:", total_roles)
        print("Data Science Roles:", data_science_roles)
        print("Percentage correct", percentage_correct, "%")

    elif "web" in posting:
        percentage_correct = web_roles / total_roles * 100 
        print("Total Roles:", total_roles)
        print("Web Dev Roles:", web_roles)
        print("Percentage correct", percentage_correct, "%")





for posting in postings:
    print_out_results(posting)
