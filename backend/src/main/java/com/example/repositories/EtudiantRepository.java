package com.example.repositories;

import com.example.entities.Etudiant;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface EtudiantRepository extends JpaRepository<Etudiant, Integer> {
    Etudiant findByEmail(String email);
}
