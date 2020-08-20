def number_of_lines():
    fhand = open('D:\DATA SCIENCE\IITR\Churn_Emails/mbox-short.txt')
    inp = fhand.read()
    fhand.close()
    count = 0
    for c in inp:
        if c == '\n':
            count += 1
    return count

print(number_of_lines())


def count_number_of_lines():
    with open('D:\DATA SCIENCE\IITR\Churn_Emails/mbox-short.txt') as f:
        count = 0
        for line in f:
            line = line.rstrip() # Remove new line characters from right
            if line.startswith('Subject:'):
                count = count + 1
    return count

print(count_number_of_lines())

def average_spam_confidence():
    with open('D:\DATA SCIENCE\IITR\Churn_Emails/mbox-short.txt') as f:
        count = 0
        spam_confidence_sum = 0
        for line in f:
            line = line.rstrip() # Remove new line characters from right
            if line.startswith('X-DSPAM-Confidence:'):
                var, value = line.split(':')
                spam_confidence_sum = spam_confidence_sum + float(value)
                count = count + 1
    return spam_confidence_sum/count

print(average_spam_confidence())

def find_email_sent_days():
    daysdict = {}
    dayslist = []

    with open('D:\DATA SCIENCE\IITR\Churn_Emails/mbox-short.txt') as f:
        for line in f:
            dayslist = line.split()
            if len(dayslist) > 3 and line.startswith('From'):
                day = dayslist[2]
                if day not in daysdict:
                    daysdict[day] = 1
                else:
                    daysdict[day] += 1
    return daysdict

print(find_email_sent_days())

def count_message_from_email():
    lineslist=[]
    emaildict={}
    with open('D:\DATA SCIENCE\IITR\Churn_Emails/mbox-short.txt') as f:
      for line in f:
        lineslist = line.split()
        if line.startswith('From:'):
          email=lineslist[1]
          if email not in emaildict:
            emaildict[email] = 1
          else:
            emaildict[email] += 1
    return emaildict

print(count_message_from_email())


def count_message_from_domain():
    lineslist=[]
    domaindict={}
    with open("/cxldata/datasets/project/mbox-short.txt") as f:
        for line in f:
            lineslist = line.split()
            if line.startswith('From:'):
                email=lineslist[1]
                domain = email.split('@')[1] 
                if domain not in domaindict:
                    domaindict[domain] = 1
                else:
                    domaindict[domain] += 1
    return domaindict