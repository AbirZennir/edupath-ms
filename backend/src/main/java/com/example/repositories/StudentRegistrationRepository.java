package com.example.repositories;

import com.example.entities.learning.StudentRegistration;
import com.example.entities.learning.StudentRegistrationId;
import org.springframework.data.jpa.repository.JpaRepository;

public interface StudentRegistrationRepository extends JpaRepository<StudentRegistration, StudentRegistrationId> {
}

