print("========================================")
print("       WELCOME TO THE QUIZ GAME")
print("========================================")

# Get participant's name
name = input("What is your name? ")

print(f"\nWelcome, {name}! 🎉")

# Ask if participant wants to play
play = input("Do you want to play the game? (yes/no): ").lower()

if play == "yes":

    print("\nGreat! Let's start the quiz!")
    print("Answer each question by entering A, B, C, or D.\n")

    score = 0

    # Question 1
    print("========================================")
    print("Question 1")
    print("What is the correct extension for a Python file?")
    print("A. .html")
    print("B. .py")
    print("C. .js")
    print("D. .java")

    answer = input("Your answer: ").lower()

    if answer == "b":
        print("Correct!")
        score += 1
    else:
        print("Wrong!")
        print("The correct answer is B.")

    # Question 2
    print("\n========================================")
    print("Question 2")
    print("Which function is used to display something in Python?")
    print("A. display()")
    print("B. show()")
    print("C. print()")
    print("D. output()")

    answer = input("Your answer: ").lower()

    if answer == "c":
        print("Correct!")
        score += 1
    else:
        print("Wrong!")
        print("The correct answer is C.")

    # Question 3
    print("\n========================================")
    print("Question 3")
    print("Which symbol is used to write a comment in Python?")
    print("A. //")
    print("B. <!-- -->")
    print("C. #")
    print("D. **")

    answer = input("Your answer: ").lower()

    if answer == "c":
        print("Correct!")
        score += 1
    else:
        print("Wrong!")
        print("The correct answer is C.")

    # Question 4
    print("\n========================================")
    print("Question 4")
    print("Which data type is used to store True or False?")
    print("A. String")
    print("B. Boolean")
    print("C. Integer")
    print("D. Float")

    answer = input("Your answer: ").lower()

    if answer == "b":
        print("Correct!")
        score += 1
    else:
        print("Wrong!")
        print("The correct answer is B.")

    # Question 5
    print("\n========================================")
    print("Question 5")
    print("Which keyword is used to create a function in Python?")
    print("A. function")
    print("B. func")
    print("C. define")
    print("D. def")

    answer = input("Your answer: ").lower()

    if answer == "d":
        print("Correct!")
        score += 1
    else:
        print("Wrong!")
        print("The correct answer is D.")

    # Final result
    print("\n========================================")
    print("             QUIZ FINISHED!")
    print("========================================")

    print(f"Participant: {name}")
    print(f"Your score: {score}/5")

    if score == 5:
        print("Excellent! 🏆 You got all answers correct!")
    elif score >= 3:
        print("Good job! 👍 Keep practicing!")
    else:
        print("Keep practicing! 💪 You can do better next time.")

else:
    print(f"\nNo problem, {name}! 👋")
    print("Thanks for visiting the Quiz Game!")