# Find the best match between key points in different images
# מציאת ההתאמה הטובה ביותר בין נקודות מפתח בתמונות שונות

## נאום המעלית

זיהוי והתאמה של נקודות מפתח בין תמונות שונות זוהי משימה חשובה ביישומי ראיה ממוחשבת.
במידה ויש התאמה חזקה בין נקודות המפתח בתמונות השונות, ניתן בקלות למצוא את הקשר בין שתי התמונות ובעזרת ההתאמה נוכל לפתור המון בעיות בתחומים מגוונים, לדוגמה: בניית מודלים תלת ממדיים, מיקום על סמך תמונה ויצירת תמונה פנורמית.
בעולם האמיתי קיימות תמונות רבות של מיקומים זהים מזווית שונות, תאורה שונה וקנה מידה שונה.
השינויים הקטנים הללו עלולים לגרום לבעיות זיהוי האובייקטים בתמונה למרות שבסופו של דבר הם עדיין מציגים את אותו המיקום.
מטרת הפרויקט שלנו הוא חקר מעמיק של האלגוריתמים הקיימים להתאמות נקודות מפתח על מנת לשפר את ההתאמות כך שיהיו מדויקות יותר ויעבדו על מקרי קיצון רבים.
הפרויקט שלנו מתבסס על המאמר SuperGlue[1] ומחולק לשני שלבים:
בשלב הראשון נממש אלגוריתמים קיימים להתאמת נקודות מפתח בין שתי תמונות, כאשר את נקודות המפתח נמצא באמצעות אלגוריתם SIFT.
השלב השני הוא מחקרי, ובו ננסה לשפר את האלגוריתמים.


## מבוא

ראייה ממוחשבת (Computer Vision) היא ענף מרכזי של מדעי המחשב, העוסק בעיבוד אוטומטי של תמונות ווידאו, במטרה לחלץ ולפרש מידע חזותי הטמון בהם. העוסקים בתחום מפתחים כלים מתמטיים וכלי תוכנה לעיבוד מידע חזותי. לדוגמה: חידוד של תמונה לא ממוקדת, חידוד של תמונה שצולמה תוך כדי תנועה מהירה, זיהוי עצמים, זיהוי נקודות מפתח ועוד.
הקלט החזותי יכול להיות צילומי סטילס, וידאו, צילומים ממצלמות מרובות, או נתונים תלת ממדיים.

זיהוי והתאמה של נקודות מפתח בין תמונות שונות זוהי משימה חשובה ביישומי ראיה ממוחשבת.
כיום משתמשים באלגוריתמי ההתאמה השונים בשלל תחומים, כגון: תמונה פנורמית, התמצאות של משתמש במרחב, קבלת מיקום מדויק על סמך תמונה, סריקת מבנים, מיון תמונות על סמך קטגוריות שונות(מיקומים, חפצים, אנשים ועוד).
המרכיב העיקרי של זיהוי והתאמה של הנקודות הוא איתור הנקודות, לזהות את נקודות "המפתח", נקודות העניין של התמונה. נקודות המפתח הן נקודות המייצגות את התמונה, נקודות בהן כיוון הגבול של האובייקט משתנה בפתאומי או שיש נקודת חיתוך בין שני מקטעי קצה.
לזיהוי נקודות המפתח נשתמש באלגוריתם SIFT , המחזיר את הנקודות והדיסקריפטורים שלהן.
את ההתאמה בין שתי תמונות מבצעים ע"י התאמה של נקודות המפתח בין התמונות.

בפרויקט זה התמקדנו בחקירת בעיית ההתאמה בין נקודות מפתח, באלגוריתמים הקיימים לפתרון בעיה זו ובניסיון לשפר ולשכלל אותם. בחרנו מספר אלגוריתמים מפורסמים ושימושיים בהם ניסינו להבין את החלקים החשובים במטרה לממש אותם בדרכנו על מנת לשפר את הצלחתם בפתרון בעיית ההתאמה.




## תיאור הבעיה

התאמת תמונות, מטלה כמעט מובנת מאליה אצל בני אדם, הופכת להיות קשה במיוחד עבור מחשב משום שתמונה דיגיטלית היא בעצם מטריצה של מספרים (פיקסלים). שינויים של זווית של אובייקט בתמונה, תאורה, אביזרים וכדומה יכולים לגרום לתמונה להיות שונה מאוד (מבחינת ערכי הפיקסלים) מתמונה אחרת של אותו המיקום.  למשל, בהרבה מקרים, תמונות של אותו אובייקט בתאורה אחרת, שונות יותר מבחינה מספרית מאשר תמונות של אובייקטים שונים בתאורה זהה.

הבעיה שבה נעסוק מעתה היא בעיית התאמה של נקודות מפתח בין שתי תמונות שונות.
זיהוי והתאמה של נקודות מפתח בין תמונות שונות זוהי משימה חשובה ביישומי ראיה ממוחשבת.
כיום משתמשים באלגוריתמי ההתאמה השונים בשלל תחומים, כגון: תמונה פנורמית, התמצאות של משתמש במרחב, קבלת מיקום מדויק על סמך תמונה, סריקת מבנים, מיון תמונות על סמך קטגוריות שונות(מיקומים, חפצים, אנשים ועוד).
המרכיב העיקרי של זיהוי והתאמה של הנקודות הוא איתור הנקודות, לזהות את נקודות "המפתח", נקודות העניין של התמונה. נקודות המפתח הן נקודות המייצגות את התמונה, נקודות בהן כיוון הגבול של האובייקט משתנה בפתאומי או שיש נקודת חיתוך בין שני מקטעי קצה. לזיהוי נקודות המפתח נשתמש באלגוריתם SIFT, (אלגוריתם זה נמצא ברוב המחקרים היום וגם במאמר שלנו) האלגוריתם מחזיר את הנקודות והדיסקריפטורים שלהן.

למטה, ניתן לראות שתי תמונות כאשר התמונה הימנית היא הזזה של התמונה השמאלית.
ההזזה בוצעה באמצעות מטריצת הומוגרפיה רנדומלית.
על התמונות מסומנות נקודות המפתח שחושבו באמצעות אלגוריתם SIFT.

![image](https://user-images.githubusercontent.com/71725532/174116388-21d867b5-1e3a-4421-afc2-de5b64f5d5e7.png)


בהינתן התמונות, נקודות המפתח והדיסקריפטורים שלהן הבעיה שנותרה לנו לפתור היא מציאת ההתאמה הטובה ביותר בין נקודות המפתח השונות.



## תמונות להמחשת התהליך

התמונה השמאלית היא התמונה המקורית, מימין אפשר לראות את התמונה המוזזת.
 
![image](https://user-images.githubusercontent.com/71725532/174116773-191fd9ff-43f8-45ff-9951-ecc9001f101a.png)


בתמונה מוצגת ההתאמה בין נקודות המפתח.

 
![image](https://user-images.githubusercontent.com/71725532/174116802-7f4361f9-a57c-42d4-8ec1-487c002fa851.png)

התמונה האמצעית מציגה את התמונה הימנית לאחר הזזה ע"פ מטריצת ההומוגרפיה ההופכית.
 
![image](https://user-images.githubusercontent.com/71725532/174116827-19f40d3f-0339-46fa-aade-f4ef4eee2ad7.png)


