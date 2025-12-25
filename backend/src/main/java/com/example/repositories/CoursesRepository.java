package com.example.repositories;
import org.springframework.data.jpa.repository.JpaRepository;
import com.example.entities.learning.Courses;
import com.example.entities.learning.CourseId;


public interface CoursesRepository extends JpaRepository<Courses, CourseId> {
}



