#!/usr/bin/env python
import fileinput
import csv
import json
import sys

# This prevents prematurely closed pipes from raising
# an exception in Python
from signal import signal, SIGPIPE, SIG_DFL

signal(SIGPIPE, SIG_DFL)

# The csv file might contain very huge fields,
# therefore increase the field_size_limit
csv.field_size_limit(sys.maxsize)


def is_insert(line):
    """
    Returns true if the line begins a SQL insert statement.
    """
    return line.startswith('INSERT INTO') or False


def get_values(line):
    """
    Returns the portion of an INSERT statement containing values
    """
    partition = line.partition('` VALUES ')
    return partition[0].strip('INSERT INTO `'), partition[2]

def values_sanity_check(values):
    """
    Ensures that values from the INSERT statement meet basic checks.
    """
    assert values
    assert values[0] == '('
    # Assertions have not been raised
    return True

def parse_announcement(row):
    return {
        'id': row[0],
        'text': row[1],
        'uid': row[2],
        'datetime' : row[3]
    }

def parse_category(row):
    return {
        'id': row[0],
        'parentid': row[1],
        'category': row[2],
        'slug': row[3],
        'lft': row[4],
        'rgt': row[5],
        'other_category': row[6],
        'meta_description' : row[7],
        'intro': row[8],
        'path': row[9],
        'q&a': row[10],
        'cafe': row[11]
    }

def parse_answer(row):
    return {
        'id': row[0],
        'qid': row[1],
        'uid': row[2],
        'answertext': row[3],
        'entered': row[4],
        'modified': row[5],
        'applicationid': row[6],
        'ip': row[7],
        'commentcount': row[8],
        'thumbsupcount': row[9],
        'thumbsdowncount': row[10],
        'bestanswer': row[11],
        'status': row[12],
        'emailcomments': row[13],
        'vid': row[14],
        'promoted': row[15],
        'cafe': row[16]
    }

def parse_question(row):
    return {
        'id': row[0],
        'uid': row[1],
        'cid': row[2],
        'questiontext': row[3],
        'description': row[4],
        'entered': row[5],
        'modified': row[6],
        'closed': row[7],
        'daysopen': row[8],
        'applicationid': row[9],
        'emailanswers': row[10],
        'open': row[11],
        'ip': row[12],
        'answercount': row[13],
        'bestanswer': row[14],
        'starcount': row[15],
        'popularity': row[16],
        'answerneed': row[17],
        'status': row[18],
        'autoclose_emailsent': row[19],
        'autoclosed': row[20],
        'vid': row[21],
        'commentcount': row[22],
        'emailcomments': row[23],
        'questquestionid': row[24],
        'private': row[25],
        'promoted': row[26],
        'cafe': row[27],
        'parentid': row[28],
        'replycount': row[29]
    }

def parse_qunread(row):
    return {
        'id': row[0],
        'uid': row[1],
        'qid': row[2],
        'aid': row[3],
        'cid': row[4],
        'lastseen': row[5],
        'entered': row[6],
        'type': row[7]
    }

def parse_user(row):
    return {
        'id': row[0],
        'username': row[1],
        'guest': row[2],
        'displayname': row[3],
        'entered': row[4],
        'modified': row[5],
        'url': row[6],
        'about': row[7],
        'lastlogin': row[18],
        'points': row[21],
        'answerrating': row[22],
        'moderator': row[23],
        'status': row[24],
        'interests_profile': row[25],
        'calculate_interests_profile': row[26],
        'notification_send': row[27],
        'notification_lastcheck': row[28],
        'bestanswer_send': row[29],
        'referrer': row[30],
        'disablevote': row[31],
        'tid': row[36],
        'medalcount': row[37]
    }

def parse_values(values, key):
    """
    Given a file handle and the raw values from a MySQL INSERT
    statement, write the equivalent CSV to the file
    """
    latest_row = []

    reader = csv.reader([values], delimiter=',',
                        doublequote=False,
                        escapechar='\\',
                        quotechar="'",
                        strict=True
                        )

    transactions = []
    buff = False
    for reader_row in reader:
        #print('READER_ROW',reader_row)
        for column in reader_row:
            # If our current string is empty...
            if len(column) == 0:
                latest_row.append('""')
                continue
            # If our string starts with an open paren
            if column[0] == "(":
                # Assume that this column does not begin
                # a new row.
                new_row = False
                # If we've been filling out a row
                if len(latest_row) > 0:
                    # Check if the previous entry ended in
                    # a close paren. If so, the row we've
                    # been filling out has been COMPLETED
                    # as:
                    #    1) the previous entry ended in a )
                    #    2) the current entry starts with a (
                    if latest_row[-1][-1] == ")":
                        # Remove the close paren.
                        latest_row[-1] = latest_row[-1][:-1]
                        new_row = True
                # If we've found a new row, write it out
                # and begin our new one
                if new_row:
                    if buff:
                        latest_row[0] = '(' + latest_row[0]
                        latest_row = buff + latest_row
                    if key.strip() == 'announcement':
                        transactions.append(parse_announcement(latest_row))
                    elif key.strip() == 'answer':
                        transactions.append(parse_answer(latest_row))
                    elif key.strip() == 'question':
                        # if buff:
                        #     latest_row = buff
                        try:
                            transactions.append(parse_question(latest_row))
                            if buff:
                              #  print('Fixed',transactions[-1])
                                buff = False
                        except:
                            #print('Could not parse question', latest_row,'COLUMN',column)
                            buff = latest_row
                            buff[-1] += ')'
                    elif key.strip() == 'question_unread':
                        transactions.append(parse_qunread(latest_row))
                    elif key.strip() == 'category':
                        transactions.append(parse_category(latest_row))
                    else:
                        print("no parse module for key",key)
                    # elif key.strip() == 'user':
                    #     json.row = parse_user(latest_row)
                    latest_row = []
                    # If we're beginning a new row, eliminate the
                    # opening parentheses.
                if len(latest_row) == 0:
                    column = column[1:]
            # Add our column to the row we're working on.
            latest_row.append(column)
        # At the end of an INSERT statement, we'll
        # have the semicolon.
        # Make sure to remove the semicolon and
        # the close paren.
        if latest_row[-1][-2:] == ");":
            latest_row[-1] = latest_row[-1][:-2]
            if key.strip() == 'announcement':
                transactions.append(parse_announcement(latest_row))
            elif key.strip() == 'answer':
                transactions.append(parse_answer(latest_row))
            elif key.strip() == 'question':
                try:
                    transactions.append(parse_question(latest_row))
                except:
                    print('Could not parse questionrow',latest_row)
            elif key.strip() == 'question_unread':
                transactions.append(parse_qunread(latest_row))
            elif key.strip() == 'category':
                transactions.append(parse_category(latest_row))
            elif key.strip() == 'user':
                transactions.append(parse_user(latest_row))

    return transactions


def main():
    """
    Parse arguments and start the program
    """
    # Iterate over all lines in all files
    # listed in sys.argv[1:]
    # or stdin if no args given.
    currentkey = ''
    tr_out = []
    unknown = False
    try:
        for line in fileinput.input(openhook=fileinput.hook_encoded('iso-8859-1')):
            # Look for an INSERT statement and parse it.
            if is_insert(line):
                key, values = get_values(line)
                if values_sanity_check(values):
                    if key != currentkey:
                        if len(tr_out) > 0:
                            print('writing output for key',currentkey)
                            with open(currentkey + '_parsed.json','w',encoding='utf-8') as out:
                                json.dump(tr_out,out)
                            tr_out = []
                        currentkey = key
                        if currentkey not in ['announcement','answer','question','question_unread','category','user']:
                            print('unknown key',currentkey)
                            unknown = True
                        else:
                            unknown = False
                    if not unknown:
                        tr_out.extend(parse_values(values, key))

    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
