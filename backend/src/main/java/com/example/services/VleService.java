package com.example.services;

import com.example.entities.learning.Vle;
import com.example.repositories.VleRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class VleService {

    @Autowired
    private VleRepository vleRepository;

    public List<Vle> getAllVles() {
        return vleRepository.findAll();
    }
}
