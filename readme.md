# Web Attack Detection Using Machine Learning :-
This is a Web Attack Detection Website made using HTML, CSS, Django and Machine Learning Algorithm.

# Note: 
Download Dataset in "/static/dataset/" Folder, Instructions given in that Directory.

# How to Create Project in Django :-
1. activate conda environment using 'activate ml'

2. go to that folder where you want to create project

3. create new project using this command 'django-admin startproject mlproject'

4. to run server use command 'python manage.py runserver'

5. run command to create an app using command 'python manage.py startapp myapp'

6. structure of an app
	a. __init__.py : tells python that your events app is package
	b. admin.py : is where you register  your apps models with sjango admin application
	c. apps.py : is a configuration file common to all django apps
	d. models.py : is where the models for your app are located
	e. tests.py : contains test procedures that will be run when testing your app
	f. views.py : views for your app are located

7. in mlproject folder add urls in urls.py as 
	'from myapp.views import hello
	 from myapp.views import login
		urlpatterns = [
    					path('admin/', admin.site.urls),
    					url(r'^hello/$',hello),
    					url(r'^login/$',login),
			      	  ]'

8. in myapp folder add function in app(views.py) and that function can be called in project(urls.py)

9. create new directory named template and static

10. now add these lines in project(setting.py) 
	'a. TEMPLATE_DIR = os.path.join(BASE_DIR,"template")
	 b. STATIC_DIR = os.path.join(BASE_DIR,"static")'
	    after BASE_DIR in project(setting.py)

11. replace this in project(setting.py) in template section ''DIRS': [TEMPLATE_DIR,],' 
	where firsty was ''DIRS': [],'

12. replace this in project(setting.py) in static section      'STATIC_URL = '/static/'
								STATICFILES_DIRS=[
    											STATIC_DIR,
										 ]'
	where firstly was 'STATIC_URL = '/static/''
13. create 'index.html' file in template folder

14. add this code in template(index.html) 
   '<!DOCTYPE html>
	<html>
    		<head>
        		<meta charset="utf-8">
        		<title>First App</title>
    		</head>
    		<body>
        		<h1>Hello this is index.html</h1>
        		{{insert_me}}
    		</body>
	</html>'

15. add this code in app(views.py)
   'from django.shortcuts import render
	from django.http import HttpResponse

	def index(request):
    		my_dict={'insert_me':"Hello i am from views.py"}
    		return render(request,'index.html',context=my_dict)'

16. add new url in project(urls.py) as 'url(r'^index/$',index),'

17. to link css files and images we have to use this syntax '"{%static '_source_' %}"'

18. to link one page to another we have to use this syntax '"{% url '_function_name_' %}"'

19. to run the project, you have to run command 'python manage.py runsever' in command prompt
