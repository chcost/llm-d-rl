"""EPP's job, prototyped in Python.

Code here decides things EPP decides in production - which replica serves a
request, when to migrate one, which peer holds a reusable prefix - so the idea
can be measured before it is written as an EPP plugin. It is staging, not a
substitute: anything that works here should graduate into EPP.
"""
