window.createTechniquesTable = function(spans) {
  const table = document.createElement("table");
  const headerRow = document.createElement("tr");
  const headers = ["Text", "Detected Techniques"];
  headers.forEach((header) => {
    const th = document.createElement("th");
    th.textContent = header;
    headerRow.appendChild(th);
  });
  table.appendChild(headerRow);

  spans.forEach((span) => {
    const row = document.createElement("tr");
    const textCell = document.createElement("td");
    textCell.textContent = span.text;
    const techniqueCell = document.createElement("td");
    const techniqueArray = span.techniques.map((item) => 
      `${item.technique.replaceAll("_", " ")} (${(item.probability * 100).toFixed(1)}%)`
  );
    techniqueCell.textContent = techniqueArray.join(", ");
    row.appendChild(textCell);
    row.appendChild(techniqueCell);
    table.appendChild(row);
  });

  return table;
};
