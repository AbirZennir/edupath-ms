package com.example.repositories;

import com.example.entities.learning.StudentRegistration;
import com.example.entities.learning.StudentRegistrationId;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;

public interface StudentRegistrationRepository extends JpaRepository<StudentRegistration, StudentRegistrationId> {

    @Query("SELECT sr FROM StudentRegistration sr WHERE sr.studentRegistrationId.id_student = :studentId")
    List<StudentRegistration> findByStudentId(@Param("studentId") int studentId);
}
