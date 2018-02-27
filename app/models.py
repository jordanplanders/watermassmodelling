from app import db
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from app import login
# from sqlalchemy.dialects.postgressql import JSON

class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    plots = db.relationship('Plot', backref='author', lazy='dynamic')
    posts = db.relationship('Post', backref='author', lazy='dynamic')


    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return '<User {}>'.format(self.username) 


class Plot(db.Model):
    __tablename__ = "plots"
    id = db.Column(db.Integer, primary_key=True)
    body = db.Column(db.String(140))
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    basemap = db.Column(db.LargeBinary, index = True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    

    def __repr__(self):
        return '<Plot {}>'.format(self.body)


class Post(db.Model):
    __tablename__ = "posts"
    id = db.Column(db.Integer, primary_key=True)
    body = db.Column(db.String(140))
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))

    def __repr__(self):
        return '<Plot {}>'.format(self.body)

# class Woa13_dloc(db.Model): 
#     __tablename__ = woa13_dlocs
class woa13(db.Model): 
    __tablename__ = "woa13s"
    id = db.Column(db.Integer, primary_key=True)
    index = db.Column(db.BigInteger, index=True)
    station = db.Column(db.String)
    longitude = db.Column(db.Float(53))
    latitude = db.Column(db.Float(53))
    depth = db.Column(db.Float(53))
    temperature = db.Column(db.Float(53))
    salinity = db.Column(db.Float(53))
    oxygen = db.Column(db.Float(53))
    oxygen_saturation = db.Column(db.Float(53))
    aou = db.Column(db.Float(53))
    phosphate = db.Column(db.Float(53))
    nitrate = db.Column(db.Float(53))

    def __repr__(self):
        return '<Plot {}>'.format(self.body)


@login.user_loader
def load_user(id):
    return User.query.get(int(id))