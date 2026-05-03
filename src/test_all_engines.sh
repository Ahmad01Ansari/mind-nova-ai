echo "--- Testing STRESS ---"
curl -s -X POST http://localhost:8000/analyze/stress \
     -H "Content-Type: application/json" \
     -H "X-Bridge-Secret: secure_internal_vpc_key_mindnova_9823" \
     -d '{"userId": "test", "mood_current": 4, "sleep_hours": 5, "workload_level": 8}' | jq .result.status

echo "--- Testing BURNOUT ---"
curl -s -X POST http://localhost:8000/analyze/burnout \
     -H "Content-Type: application/json" \
     -H "X-Bridge-Secret: secure_internal_vpc_key_mindnova_9823" \
     -d '{"userId": "test", "work_hours": 10, "sleep_hours": 5, "break_frequency": 1}' | jq .result.status

echo "--- Testing ANXIETY (Hybrid) ---"
curl -s -X POST http://localhost:8000/analyze/anxiety \
     -H "Content-Type: application/json" \
     -H "X-Bridge-Secret: secure_internal_vpc_key_mindnova_9823" \
     -d '{"userId": "test", "gad2_score": 8, "phq2_score": 4, "sleep_hours": 4, "academic_stress": 9, "social_activity": 2}' | jq .result

echo "--- Testing DEPRESSION (Hybrid) ---"
curl -s -X POST http://localhost:8000/analyze/depression \
     -H "Content-Type: application/json" \
     -H "X-Bridge-Secret: secure_internal_vpc_key_mindnova_9823" \
     -d '{"userId": "test", "phq2_score": 8, "gad2_score": 4, "sleep_hours": 4, "academic_stress": 9}' | jq .result

echo "--- Testing DETERIORATION ---"
curl -s -X POST http://localhost:8000/analyze/deterioration \
     -H "Content-Type: application/json" \
     -H "X-Bridge-Secret: secure_internal_vpc_key_mindnova_9823" \
     -d '{
       "userId": "test",
       "history": [
         {"day": 1, "mood": 8, "sleep": 8, "workload": 2},
         {"day": 2, "mood": 7, "sleep": 7, "workload": 3},
         {"day": 3, "mood": 6, "sleep": 6, "workload": 4},
         {"day": 4, "mood": 5, "sleep": 5, "workload": 5},
         {"day": 5, "mood": 4, "sleep": 4, "workload": 6},
         {"day": 6, "mood": 3, "sleep": 3, "workload": 7},
         {"day": 7, "mood": 2, "sleep": 2, "workload": 8}
       ]
     }' | jq .result.status
