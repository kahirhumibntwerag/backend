from models import User, engine, Address
from sqlalchemy.orm import sessionmaker
import random

Session = sessionmaker(bind=engine)
session = Session()

users = session.query(User).where(User.name=='John').all()
for user in users:
    print(user.name, user.age)
    print(user.address[0].city, user.address[0].zip_code, user.address[0].state)





