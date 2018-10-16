#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 16:52:00 2018

@author: channerduan
"""

#%% Assertion
# standard example
def apply_discount(product, discount):
    price = int(product['price'] * (1.0 - discount))
    assert 0 <= price <= product['price']
    return price
shoes = {'name': 'Fancy Shoes', 'price': 14900}
apply_discount(shoes, 0.25)

# it is just for debug, do not do data validation(assert can be skip and cause security problem)

# Besides, there is pitfall below (which never fail):
assert(1 == 2, 'This should fail')  # it is a negative example!!!

#%% Long str auto concatenate
my_str = ('This is a super long string constant '
          'spread out across multiple lines. '
          'And look, no backslash characters needed!'
          )
print my_str

#%% Complacent Comma Placement
# You add extra comma in list/dict/set, which makes you code clean and easier to maintain
names = [
        'Alice',
        'Bob',
        'Dilbert',
        'Jane', 
        ]
print names

#%% Feature 'with' in pythn
# eg.1
class ManagedFile:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.file = open(self.name, 'w') 
        return self.file
    def __exit__(self, exc_type, exc_val, exc_tb): 
        if self.file:
            self.file.close()

with ManagedFile('hello.txt') as f:
    f.write('hello, world!')
    f.write('bye now!!!')

# eg.2
class Indenter:
    def __init__(self):
        self.level = 0
    def __enter__(self): 
        self.level += 1
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.level -= 1
    def print_(self, text):
        print('    ' * self.level + text)
    
with Indenter() as indent: 
    indent.print_('hi!') 
    with indent:
        indent.print_('hello') 
        with indent:
            indent.print_('bonjour')
    indent.print_('hey')
    
    
#%% Specific variable names (including 5 different styles)
##
#
## • Single Leading Underscore: _var
#       hints private(internal use) variable or method of some class. 
#       It is just convention~
## • Single Trailing Underscore: var_
#       avoid naming conflicts with Python keywords, such as print_().
#       It is just convention~
## • Double Leading Underscore: __var
#       Python interpreter automatically rename(mangling) this kind of variables/methods to avoid collision. 
#       It is forced! experienced Pythonistas call it dunder(double underscore)~
# eg.1
class Test:
    def __init__(self):
        self.foo = 11
        self._bar = 23
        self.__baz = 42
t = Test()
print dir(t)      # you see the variable __baz is renamed
# eg.2
class MangledMethod: 
    def __method(self):
        return 42222222
    def call_it(self):
        return self.__method()
print MangledMethod().call_it()
#print MangledMethod().__method()    # you see crash
# eg. 3
_MangledGlobal__mangled = 123123123123
class MangledGlobal: 
    def test(self):
        return __mangled
print MangledGlobal().test()
## • DoubleLeadingandTrailingUnderscore:__var__ 
#       Special use in the language, such as __init__ (constructor) or __call__

## • Single Underscore: _
#       Temporary variable or used to unpack

for _ in range(6):
    print('Hello, World.')
car = ('red', 'auto', 12, 3812.4)
color, _, _, mileage = car
print mileage


#%% String Formatting (3 style)
## • old style
name = 'Bob'
errno = 50159747054
print 'Hello, %s' % name
print '%x' % errno
print 'Hey %s, there is a 0x%x error!' % (name, errno)
# dict~ you donot need to care the order of params~
print 'Hey %(name)s, there is a 0x%(errno)x error!' % {"name": name, "errno": errno }

## • new style
print 'Hello, {}'.format(name)
print 'Hey {name}, there is a 0x{errno:x} error!'.format(name=name, errno=errno)

## • Template style
from string import Template
templ_string = 'Hey $name, there is a $error error!'
print Template(templ_string).substitute(name=name, error=hex(errno))

#%% Easter Egg of Python~
import this
help(this)


#%% Function of Python! Deep Insight!!! part 1
# standard function here
def yell(text):
    return text.upper() + '!'
print yell('hello')
# all functions are objects!!! it can be assigned to another variable
bark = yell
print bark('woof')
# del only delete the name of function rather than the function itself
# A variable pointing to a function and the function itself are really two separate concerns.
del yell
#print yell('hello?')
print bark('hey')
# function can be stored
funcs = [bark, str.lower, str.capitalize]
print funcs
for f in funcs:
    print (f, f('hey there'))
funcs[0]('heyho')

#%% Function of Python! Deep Insight!!! part 2
## functions can be parameters
def greet(func):
    greeting = func('Hi, I am a Python program')
    print(greeting)
greet(bark)
def whisper(text):
    return text.lower() + '...'
greet(whisper)
# Functions that can accept other functions as arguments are also called higher-order functions. 
# They are a necessity for the functional pro- gramming style.
# map is a built in higher-order function
print map(bark, ['hello', 'hey', 'hi'])

#%% Function of Python! Deep Insight!!! part 3
## functions can be returned (nested function)
# it likes factory pattern
def get_speak_func(volume): 
    def whisper(text):
        return text.lower() + '...' 
    def yell(text):
        return text.upper() + '!' 
    if volume > 0.5:
        return yell 
    else:
        return whisper

print get_speak_func(0.3)
print get_speak_func(0.7)
speak_func = get_speak_func(0.1)
print speak_func('Hello')

## with parameter(local state) from parent function(called lexical closure!!!)
## Functions do this are called closure!
# A closure remembers the values from its enclosing lexical scope even when the program flow is no longer in that scope.
def get_speak_func(text, volume):
    def whisper():
        return text.lower() + '...'
    def yell():
        return text.upper() + '!'
    if volume > 0.5:
        return yell
    else:
        return whisper
print get_speak_func('Hello, World', 0.5001)()

# factory make_adder create and config specific function plus_3/plus_5
def make_adder(n): 
    def add(x):
        return x + n
    return add
plus_3 = make_adder(3)
plus_5 = make_adder(5)

print plus_3(4)
print plus_5(4)

#%% Function of Python! Deep Insight!!! part 4
## Functions are objects in Python while objects aren't functions.
## But we can make objects callable!
class Adder:
    def __init__(self, n):
        self.n = n
    def __call__(self, x):
        return self.n + x

plus_3 = Adder(3)
print plus_3(4)

# check callable for objects
print callable(plus_3)
print callable('I am a string~')

#%% Lambda (it is so cute)
## A shortcut for declaring small anonymous functions. anonymous! anonymous! anonymous!
# it is also called single expression function (restricted to only own one function)
add = lambda x, y: x + y
print 'simple lambda:', add(5, 3)
# just the same with regular way using def
def add(x, y):
    return x + y
print 'regular function:', add(5, 3)
# This case is really anonymous, there is no name for the function
print 'real anonymous:', (lambda x, y: x + y)(5, 3)

## Some frequent use cases
# eg.1
tuples = [(1, 'd'), (2, 'b'), (4, 'a'), (3, 'c')]
print sorted(tuples, key=lambda x: x[0], reverse=True)
# eg.2
print sorted(range(-5, 6), key=lambda x: x * x)

## Lambdas with lexical closures
def make_adder(n):
    return lambda x: x + n
plus_3 = make_adder(3)
plus_5 = make_adder(5)
print plus_3(4)
print plus_5(4)

## Take care of lambda~ It may make codes confusing.
# donot use too many or too complex lambda.
# bad case in generator expression, it is bad! bad! bad!
print list(filter(lambda x: x % 2 == 0, range(16)))
# cleaner expression is here without lambda
# a list comprehension offer more clarity
print [x for x in range(16) if x % 2 == 0]




#%% Decorators. part 1
# extend and modify the behavior of a callable (functions, methods, and classes)
# understanding decorators is a milestone for any serious Python programmer

## Simplest decorator
# get a function input and give another function output
def null_decorator(func): return func
def greet(): return 'Hello!'
greet = null_decorator(greet)
print greet()
## syntactic sugar!
@null_decorator
def greet(): return 'Hello!'
print greet()

## Real case of decorator
def uppercase(func): 
    def wrapper():
        original_result = func()
        modified_result = original_result.upper() 
        return modified_result
    return wrapper
@uppercase
def greet(): return 'Hello!'
print greet()

## Multiple decorators
# apply in bottom to top order; one thing keeps in mind if performance problem
def strong(func): 
    def wrapper():
        return '<strong>' + func() + '</strong>' 
    return wrapper
def emphasis(func): 
    def wrapper():
        return '<em>' + func() + '</em>' 
    return wrapper

@strong
@emphasis
def greet(): return 'Hello!'
print greet()


#%% Decorators. part 2
## Decorating functions with parameters(input arguements)!!! honestly, it is common.
# args is a tuple of positional args, while kwargs is a dict of keyword args
# trace example, simple but really useful
def trace(func):
    def wrapper(*args, **kwargs):
        print('TRACE: calling %s() with %s, %s' %(func.__name__, args, kwargs))
        original_result = func(*args, **kwargs)
        print('TRACE: %s() returned %s' %(func.__name__, original_result))
        return original_result 
    return wrapper

@trace
def say(name, line): return '%s: %s' %(name,line)
print say('Jane', 'Hello, World')

## Decorator with metadata
# decorator lose metadata in normal case
def uppercase(func): 
    def wrapper():
        return func().upper() 
    return wrapper
def greet():
    """Return a friendly greeting.""" 
    return 'Hello!'
decorated_greet = uppercase(greet)
print greet.__name__
print greet.__doc__
print decorated_greet.__name__
print decorated_greet.__doc__
# functools.wraps is a way to fix it
import functools
def uppercase(func): 
    @functools.wraps(func) 
    def wrapper():
        return func().upper() 
    return wrapper
@uppercase
def greet():
    """Return a friendly greeting.""" 
    return 'Hello!'
print greet.__name__
print greet.__doc__
# you would better use functools for all of your decorators~

#%% Function definition args & kwargs
# args and kwargs here are optional arguements in function definition
# slightly different with decorator definition (no required parameter, all parameters transformed into args and kwargs)
def foo(required, *args, **kwargs): 
    print(required)
    if args: print(args)
    if kwargs: print(kwargs)
# bad case
#foo()
# right case
foo('hello')
foo('hello', 1, 2, 3)
foo('hello', 1, 2, 3, key1='value', key2=999)


#%% Function arguements unpacking
# simple way to run function with a list of parameters~ (this skill is really useful in scala with spark)
# only a * is for tuple/list parameters, while ** for dict parameters
# putting a * before an iterable in a function call will unpack it and pass its elements as separate positional arguments to the called function
def print_vector(x, y, z):
    print('<%s, %s, %s>' % (x, y, z))

tuple_vec = (1, 0, 1)
list_vec = [1, 0, 1]
dict_vec = {'x':1,'y':1,'z':1}
print_vector(0, 1, 0)
print_vector(*tuple_vec)
print_vector(*list_vec)
print_vector(**dict_vec)

#%% None return is default in python's function
def foo1(value): 
    if value: return value 
    else: return None
def foo2(value):
    """Bare return statement implies `return None`""" 
    if value: return value 
    else: return
def foo3(value):
    """Missing return statement implies `return None`""" 
    if value: return value

print type(foo1(0)), type(foo2(0)), type(foo3(0))



#%% Object Comparison
# here start object !
# here start object !
# here start object !
# here start object !
#
# kind of similar with java, ==(equal in java) and is(== in java) are different
a = [1, 2, 3]
b = a
c = list(a)
print a == b, a is b
# the content of a and c are same, but they point to different objects
print a == c, a is c

#%% Object String Conversion
# __str__ and __repr__ are python features that specificly used in this language
class Car:
    def __init__(self, color, mileage):
        self.color = color 
        self.mileage = mileage
    def __repr__(self):
        return '__repr__ for Car'
    def __str__(self):
        return '__str__ for Car'

my_car = Car('red', 37281)
# print or single str use the result of __str__
print my_car
print str(my_car)
# containers like lists and dicts use the result of __repr__
print str([my_car])
# inspecting uses the result of __repr__
my_car

#%%

import datetime
today = datetime.date.today()


#%%

ll = ['aa999', 'aaZ', 'aa-', 'aa_', 'aa~', 'aa!', 'aa1']

print sorted(ll)


#%%
print char('_')






#%%









#%%









#%%









#%%




















