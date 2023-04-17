# NYC_Trip

<b>! NYC_Trip veri seti kullanılarak yolcuların bindiği ve ineceği lokasyonları seçerek gidecekleri mesefanin süresini hesaplayan bir projedir. !</br></b>
<b>! Projenin ana amacı bir modeli canlıya alacak gibi simülasyon yapmaktır. Model bir veri kaynağından gelen verileri okuyup her 5dk da bir modeli tekrardan eğitip bir önceki model ile kıyaslayarak en başarılı modeli canlıya alır. Bu model ile kullanıcılar arayüz vasıtası ile tahminleme işlemini yapabilirler.!</br></b>

<b>! Projenin çalıştırılabilmesi içine gerekli teknolojiler: !</br></b>
* Docker  </br>
* Hadoop  </br>

<b>! requirements dosyasında gerekli kütüphanler ve versiyonları verilmiştir !</br></b>

<b>Projede kullanılan teknolojiler: </br></b>
* Docker: KUllanılan teknolojileri her ortamda çalışabilmesini sağlamak için kullanılmıştır. Bütün teknolojiler docker konteynırı haline getirilerek kullanılmıştır. </br>
* Streamlit: Oluşturulan modeli web ortamında sunabilmek için kullanılmıştır.</br>
* Prefect: Modelin belli aralıklarla otomatik olarak çalıştırılıp eğitilmesi için kullanılmıştır. </br>
* MLflow: Model parametrelerini loglamak ve modeli kayıt etmek için kullanılmıştır.</br>
* Hadoop: MLflow ile modeli hadoop hdfs sistemine kayıt etmek için kullanılmıştır.</br>
* PostgreSql: MLflow ile model parametrelerini kayıt etmek için kullanılmıştır.</br>
* Xgboost: Model Xgboost algoritması kullanılarak oluşturulmuştur.</br>
* Hyperopt: Model hyperparametrelerini optimize etmek için kullanılmıştır.</br>
* Pandas: Veri ön işleme için kullanılmıştır.</br>

<b>Projenin Çalıştırılabilmesi için gerekli adımlar: </br></b>
* "./ docker compose up" </br>
  * [localhost:5000](http://localhost:5000/) for MLflow UI
  ![mlflow](https://user-images.githubusercontent.com/43652313/232592007-2b000ad8-8e87-439a-a549-c655f3a6789b.png)
  * [localhost:8080](http://localhost:8080/) for PostgreSQL UI
  ![postgresql](https://user-images.githubusercontent.com/43652313/232592062-908f8c23-e8fb-486a-b561-6327a5f86b20.png)

* "./hadoop/ docker compose up" </br>
  * [localhost:9870](http://localhost:9870/) for Hadoop HDFS UI
  * ![hadoop](https://user-images.githubusercontent.com/43652313/232592128-cdc49622-de19-4417-be8d-0d17aa1d78a9.png)

* "./prefect/ docker compose up" </br>
  * [localhost:4200](http://localhost:4200/) for Prefect UI
  * ![prefect](https://user-images.githubusercontent.com/43652313/232592546-310cbe8a-004c-4628-86e6-881110bc6cf6.png)

* "./ streamlit run uı.py" </br>
  * [localhost:8501](http://localhost:8501/) for Streamlit UI
  * ![streamlit](https://user-images.githubusercontent.com/43652313/232594601-41a3d7d3-6878-4b85-8963-09cf39f33e0e.png)

