package com.example.repositories;

import com.example.entities.StudentRiskPrediction;
import org.springframework.data.jpa.repository.JpaRepository;

public interface StudentRiskPredictionRepository extends JpaRepository<StudentRiskPrediction, Long> {
}
