from flask_wtf import FlaskForm
from wtforms import StringField, SelectMultipleField, PasswordField, BooleanField, RadioField, SubmitField, SelectField
from wtforms.validators import ValidationError, DataRequired, Email, EqualTo
from app.models import User


class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')


class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    password2 = PasswordField(
        'Repeat Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user is not None:
            raise ValidationError('Please use a different username.')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user is not None:
            raise ValidationError('Please use a different email address.')
# any methods that match the pattern validate_<field_name>, WTForms takes those as custom validators

class PlotForm(FlaskForm):
    tracer = SelectMultipleField('Tracers', choices=[('oxygen', 'Oxygen'), ('phosphate', 'Phosphate'), ('salinity', 'Salinity')], validators=[DataRequired()])
    plot_type = RadioField('Plot Type', choices = [('plan', 'Cross Section'), ('EWsection', 'Depth Slice (East-West line)'), ('NSsection', 'Depth Slice (North-South line)')],validators=[DataRequired()] )
    lat_n = StringField('Lat-northern bound')
    lat_s = StringField('Lat-southern bound')
    lon_w = StringField('Lon-western bound')
    lon_e =StringField('Lon-eastern bound')
    depth= StringField('Depth (depth slice only)')
    # raw = RadioField('Data Visualization Type', choices = [('raw', 'Raw')], validators=[DataRequired()])
    # model = RadioField('Data Visualization Type', choices = [('model', 'Model')], validators=[DataRequired()])
    raw_model = RadioField('Data Visualization Type', choices = [('raw', 'Raw'), ('model', 'Model')], validators=[DataRequired()])
    model_opts = SelectMultipleField('Model Options', choices=[('MiniBatchKMeans', 'Mini Batch KMeans'), ('AgglomerativeClustering', 'Agglomerative Clustering'), ('SpectralClustering', 'Spectral Clustering'), ('AffinityPropagation', 'Affinity Propagation'), ('DBSCAN', 'DBScan'), ('Ward', 'Ward')])
    sil_coef = RadioField('Sillohuette Coef Plot', choices=[('yes', 'yes'), ('no', 'no')])
    submit = SubmitField('Plot')

    # def validate_raw(self, raw, model):
    #     if raw.data == model.data:
    #         raise ValidationError('Please choose between "Raw" or "Model"')

    # def validate_model(self, raw, model):
    #     if raw.data == model.data:
    #         raise ValidationError('Please choose between "Raw" or "Model"')

    