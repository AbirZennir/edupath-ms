from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from database import Base

class StudentInfo(Base):
    __tablename__ = "student_info"
    # Assuming standard OULAD schema columns, guessing primary key or composite PK
    # OULAD usually has (code_module, code_presentation, id_student) as composite PK.
    # We'll use id_student as a proxy for simple querying or map strictly.
    
    id_student = Column(Integer, primary_key=True)
    code_module = Column(String(50), primary_key=True)
    code_presentation = Column(String(50), primary_key=True)
    gender = Column(String(10))
    region = Column(String(50))
    highest_education = Column(String(50))
    imd_band = Column(String(20))
    age_band = Column(String(20))
    num_of_prev_attempts = Column(Integer)
    studied_credits = Column(Integer)
    disability = Column(String(10))
    final_result = Column(String(20)) 

class StudentAssessment(Base):
    __tablename__ = "student_assessment"
    id_assessment = Column(Integer, primary_key=True)
    id_student = Column(Integer, primary_key=True)
    date_submitted = Column(Integer)
    is_banked = Column(Integer)
    score = Column(Float)

class StudentVle(Base):
    __tablename__ = "student_vle"
    code_module = Column(String(50), primary_key=True)
    code_presentation = Column(String(50), primary_key=True)
    id_student = Column(Integer, primary_key=True)
    id_site = Column(Integer, primary_key=True)
    date = Column(Integer, primary_key=True)
    sum_click = Column(Integer)

class Vle(Base):
    __tablename__ = "vle"
    id_site = Column(Integer, primary_key=True)
    code_module = Column(String(50))
    code_presentation = Column(String(50))
    activity_type = Column(String(50))
    week_from = Column(Integer)
    week_to = Column(Integer)
