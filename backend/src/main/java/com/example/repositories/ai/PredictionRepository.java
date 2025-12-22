package com.example.repositories.ai;


import com.example.entities.ai.Prediction;
import com.example.entities.ai.PredictionId;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface PredictionRepository extends JpaRepository<Prediction, PredictionId> {

    /**
     * Find all students at risk (prediction = 1)
     */
    List<Prediction> findByPrediction(Integer prediction);

    /**
     * Find top N at-risk students ordered by risk score descending
     */
    List<Prediction> findTop10ByPredictionOrderByRiskScoreDesc(Integer prediction);

    /**
     * Count students at risk
     */
    long countByPrediction(Integer prediction);

    /**
     * Get average risk score across all predictions
     */
    @Query("SELECT AVG(p.riskScore) FROM Prediction p")
    Double getAverageRiskScore();
}
