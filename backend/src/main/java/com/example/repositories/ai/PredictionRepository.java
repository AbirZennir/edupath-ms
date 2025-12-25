package com.example.repositories.ai;


import com.example.entities.ai.Prediction;
import com.example.entities.ai.PredictionId;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface PredictionRepository extends JpaRepository<Prediction, PredictionId> {

    List<Prediction> findByPrediction(Integer prediction);
    List<Prediction> findTop10ByPredictionOrderByRiskScoreDesc(Integer prediction);

    long countByPrediction(Integer prediction);
    @Query("SELECT AVG(p.riskScore) FROM Prediction p")
    Double getAverageRiskScore();
}
