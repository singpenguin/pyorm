#!/usr/bin/env python
# encoding:utf-8

import pyorm


pyorm.database("mysql", db="", user="root", passwd="root", charset="utf8")
#pyorm.database("sqlite3", db="test")


class User(pyorm.Model):
    #table_name = "user"
    id = pyorm.Field()
    name = pyorm.Field()

if __name__ == "__main__":
    user = User.findone(id=1)
    print(user.id, user.name)
