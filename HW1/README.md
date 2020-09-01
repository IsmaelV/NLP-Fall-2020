# Homework 1: Full Name Predictor
All of README file written by Ron Artstein unless otherwise noted (last section of README).
## Due date: TBD
This assignment counts for 5% of the course grade. Assignments turned in after the deadline are subject to a grade 
penalty (the precise penalty will be announced together with the deadline).

### Overview
Person names in the English language typically consist of one or more forenames followed by one or more surnames 
(optionally preceded by zero or more titles and followed by zero or more suffixes). This situation can create ambiguity, 
as it is often unclear whether a particular name is a forename or a surname. For example, given the sequence Imogen and 
Andrew Lloyd Webber, it is not possible to tell what the full name of Imogen is, since that would depend on whether 
Lloyd is part of Andrew’s forename or surname (as it turns out, it is a surname: Imogen Lloyd Webber is the daughter of 
Andrew Lloyd Webber). This exercise explores ways of dealing with this kind of ambiguity.

You will write a program that takes a string representing the names of two persons (joined by and), and tries to predict 
the full name of the first person in the string. To develop your program, you will be given a set of names with correct 
solutions: these are not names of real people – rather, they have been constructed based on lists of common forenames 
and surnames. The names before the and are the first person’s forenames, any titles they may have, and possibly surnames; 
the names after the and are the second person’s full name. For each entry, your program will output its best guess as to 
the first person’s full name. The assignment will be graded based on accuracy, that is the number of names predicted 
correctly on an unseen data set constructed the same way.

### Data
A set of development and test data is available as a compressed ZIP archive on Blackboard. The uncompressed archive 
contains the following files:
* A test file with a list of conjoined names.
* A key file with the same list of conjoined names, and the correct full name of the first person for each example.
* Three lists of name frequencies from U.S. census data (these lists are available on the web, and included in the 
package for your convenience).
    * [Frequency of female first names from the 1990 census](https://www2.census.gov/topics/genealogy/1990surnames/dist.female.first)
    * [Frequency of male first names from the 1990 census](https://www2.census.gov/topics/genealogy/1990surnames/dist.male.first)
    * [Frequency of surnames from the 2010 census](https://www2.census.gov/topics/genealogy/2010surnames/names.zip)
* Readme and license files, which will not be used for the exercise, but may contain useful information.

The submission script will run your program on the test file and compare the output to the key file. The grading script 
will do the same, but on a different pair of test and key files which you have not seen before.

### Program
You will write a program called `full-name-predictor.py` in Python 3 (Python 2 has been deprecated), which will take the 
path to the test file as a command-line argument. Your program will be invoked in the following way:

> python full-name-predictor.py /path/to/test/data

The program will read the test data, and write its answers to a file called full-name-output.csv. The output file must 
be in the same format of the key file.

### Submission
All submissions will be completed through [Vocareum](https://labs.vocareum.com/main/main.php); please consult the 
[instructions for how to use Vocareum.](http://ron.artstein.org/csci544-2020-08/Student-Help-Vocareum.pdf)

Multiple submissions are allowed; only the final submission will be graded. Each time you submit, a submission script is 
invoked, which runs the program on the test data. Do not include the data in your submission: the submission script 
reads the data from a central directory, not from your personal directory. You should only upload your program file to 
Vocareum, that is `full-name-predictor.py`; if your program uses auxiliary files (for example, lists of common names), 
then you must also include these in your personal directory.

You are encouraged to submit early and often in order to iron out any problems, especially issues with the format of the 
final output.

_The output of your program will be graded automatically; failure to format your output correctly may result in very low 
scores, which will not be changed._

For full credit, make sure to submit your assignment well before the deadline. The time of submission recorded by the 
system is the time used for determining late penalties. If your submission is received late, whatever the reason 
(including equipment failure and network latencies or outages), it will incur a late penalty.

### Grading
After the due date, we will run your program on a unseen test data, and compare your output to the key to that test 
data. Your grade will be the accuracy of your output, scaled to the output of a predictor developed by the instructional 
staff (so if, for example, that predictor has an accuracy of 90%, then an accuracy of 90% or above will receive full 
credit, and an accuracy of 81% will receive 90% credit).

## My Implementation (written by Ismael Villegas-Molina)
* **Preprocessing:** Read and prepare the data to make predictions
    * **Read in Data:** Read the test data as well as the frequencies for surnames and female/male forenames
    * **Prepare Data:** Split the former and latter name and do the following to each:
        * Split into tokens
        * Determine whether forename is female or male with female/male frequencies
        * Make an list for the frequency for each token, one for surname and one for the gendered name
            * If not in the frequency dictionaries, then set as -1
* **Prediction:** Using the preprocessed data, make predictions
    * **Initial Forename:** Every prediction start with the first token of the former name (e.g. Jon Paul Doe --> Jon)
    * **Former Name Forename Determination:** Get former name and determine whether each token is a forename
        * Use gendered and surname frequencies for each token. If surname_freq > gendered_freq --> False. Else --> True
        * As soon as forename isn't detected in token sequence (detected as False), then subsequent tokens also False
    * **Former Name Surname Determination:** Get former name and determine whether each token is a surname
        * Use gendered and surname frequencies for each token. If surname_freq > gendered_freq --> True. Else --> False
        * As soon as a surname is detected in token sequence (detected as True), then subsequent tokens also True
    * **If surname found in former name, _IGNORE LATTER NAME_ AND SKIP TO NEXT PREDICTION**
    * **Latter Name Surname Determination:** Get latter name and determine whether each token is a surname
        * Use gendered and surname frequencies for each token. If surname_freq > gendered_freq --> True. Else --> False
        * As soon as a surname is detected in token sequence (detected as True), then subsequent tokens also True
    * **If no surname found in latter name, _APPEND THE LAST TOKEN TO PREDICTION_ as it is guaranteed to be a surname**
    * **Single Name Prediction:** If after all other steps we still only predict one name, **_APPEND THE LAST TOKEN TO 
    PREDICTION_** as it is guaranteed to be a surname
* **Write to File:** Write all predictions in a file to feed into Vocareum