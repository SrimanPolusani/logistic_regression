import numpy as np

creds = np.array(
    [[4, 3, 2, 1, 3, 3],
     [3, 4, 3, 3, 4, 3, 1, 3, 1],
     [4, 4, 3, 3, 1, 1, 1, 3, 2],
     [2, 3, 3, 3, 1, 1, 1, 1, 4, 3, 2],
     [4, 3, 3, 3, 3, 1, 1, 1, 1, 3, 2],
     [3, 3, 3, 3, 2, 1, 1, 1, 1, 3, 2],
     [3, 3, 3, 3, 1, 1, 1, 1, 1, 3, 2],
     [8]]
)

grades = (
    [[9, 7, 8, 10, 8, 8],
     [5, 9, 7, 8, 9, 8, 10, 8, 10],
     [6, 6, 8, 8, 8, 8, 8, 8, 8],
     [8, 7, 8, 7, 8, 9, 9, 10, 7, 9, 9],
     [9, 7, 7, 8, 8, 9, 7, 9, 10, 8, 9],
     [8, 7, 8, 7, 7, 8, 8, 8, 8, 7, 7],
     [8, 8, 8, 9, 8, 9, 9, 9, 9, 8, 8],
     [9]]
)

gpa = np.zeros(8)


def calculation(all_grads, all_creds):
    for sem_no in range(8):
        numerator = 0
        for subject_no, gred_no in enumerate(all_creds[sem_no]):
            numerator += gred_no * all_grads[sem_no][subject_no]
        gpa[sem_no] = numerator / (sum(all_creds[sem_no]))


def cal_cgpa(gpa_set):
    cgpa = 0
    sum_creds = 0
    for sem_no, gpa in enumerate(gpa_set):
        cgpa += sum(creds[sem_no])*gpa
        sum_creds += sum(creds[sem_no])
    cgpa = cgpa/sum_creds
    return cgpa


calculation(grades, creds)
print(gpa)
print(cal_cgpa(gpa))
